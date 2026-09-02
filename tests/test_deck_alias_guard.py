"""Regression: the ``type_id -> %LPD_B&C alias`` translation must never be a
silent no-op on the deck-EMISSION path.

The defect (``data/reports/hgd569_degeneracy_memo_20260830.md``): a caller that
built a :class:`~lpopt.search.verify.WaveVerifier` without ``resolver=`` got an
internal fallback :class:`~lpopt.search.assets.CaseAssetResolver` whose alias
bridge was EMPTY.  ``alias_pattern`` / ``prepare_cycle1_deck`` then returned the
pattern unchanged, so ``%LPD_SHF`` carried full 13-character fuel ``type_id``\\ s
that are absent from the deck's ``%LPD_B&C``.  MASTER emits no diagnostic for an
unknown batch id — it absorbed every one of them as a single unrelated batch —
so 160 chains were computed on a core nobody designed and were labelled as if
they were the designed one.

Three properties are pinned here:

1. deck emission THROUGH ``WaveVerifier`` yields only roster aliases in
   ``%LPD_SHF`` (the fixed path);
2. :func:`~lpopt.search.assets.validate_reload_deck` REJECTS a deck whose SHF
   names a raw ``type_id`` (the guard, independent of who emitted it);
3. the pre-fix construction — a verifier for a paramA package with no explicit
   ``resolver=`` — now either translates or raises, and never silently emits.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from lpopt.search.assets import (
    CaseAssetResolver,
    DeckValidationError,
    ResolvedAssets,
    parse_lpd_bc_batch_ids,
    registry_aliases_from_package,
    validate_reload_deck,
)
from lpopt.search.genome import random_genome
from lpopt.search.stub import StubEvaluator
from lpopt.search.verify import WaveEntry, WaveVerifier
from lpopt.vendor.masterrl.domain import CaseKey
from lpopt.vendor.masterrl.master import extract_lpd_shf

# --------------------------------------------------------------------------- #
# a 3-type synth-deck fixture with MULTI-CHARACTER type ids
# --------------------------------------------------------------------------- #
#: The shape of the real defect: full parametric type_ids (13 chars), whose
#: leading two characters COLLIDE ("P6"), bridged to distinct 2-char aliases.
TYPE_IDS = ("P6253Z1G06N24", "P6253Z2G10N24", "P6257Z1G06N24")
ALIASES = ("S3", "S5", "P6")
REGISTRY = dict(zip(TYPE_IDS, ALIASES))
PAIR = "_".join(TYPE_IDS)
FEED = 125
#: ``feed = 1 + 4N`` (:func:`lpopt.search.genome.feed_from_fresh_units`).
N_FRESH = 31
DIMS = (40, 42)
RESTART = "MAS_RST.APRQ_11_0705.02"

#: A structurally complete reload deck carrying a ``%LPD_B&C`` roster of the
#: three aliases plus the reflectors — i.e. exactly the roster a paramA package
#: deck declares, in miniature.
RELOAD_DECK = (
    "%JOB_TYP\n"
    "        1       stead                               # irrst=1 (restart)\n"
    f"        {RESTART}\n"
    "        xsl     MAS_XSL\n"
    "        out     MAS_OUT\n"
    "%JOB_IDE\n"
    "        APRQ    12\n"
    "%GEN_DIM\n"
    f"        10      10      27      {DIMS[0]}      {DIMS[1]}   # nx,ny,nz,nbatch,ncomp\n"
    "%LPD_B&C\n"
    "        R1                      27*-2\n"
    "        R2                      27*-3\n"
    "        S3      -1              25*1            -5\n"
    "        S5      -1              25*2            -5\n"
    "        P6      -1              25*3            -5\n"
    "%LPD_C&X\n"
    "        1       FA_S3\n"
    "        2       FA_S5\n"
    "        3       FA_P6\n"
    "%LPD_SHF\n"
    "        F S3  0,\n"
    "%EXE_DEP                                            # BOC\n"
    "        0.0\n"
    "%EDT_OPT                                            # EOC restart write\n"
    "        1\n"
    "%END\n"
)


def _package(tmp_path: Path, *, with_registry: bool) -> Path:
    """A minimal paramA-shaped package root (only ``registry.json`` matters here)."""
    pkg = tmp_path / "package"
    pkg.mkdir(parents=True, exist_ok=True)
    if with_registry:
        (pkg / "registry.json").write_text(
            json.dumps({"aliases": REGISTRY}), encoding="utf-8"
        )
    return pkg


def _entry(tmp_path: Path, template: Path, restart: Path) -> WaveEntry:
    pattern = random_genome(random.Random(569), PAIR, N_FRESH).to_pattern()
    assert pattern.feed == FEED
    assert {i.batch for i in pattern.items if i.is_fresh} <= set(TYPE_IDS)
    return WaveEntry(
        pattern,
        CaseKey(PAIR, FEED),
        ResolvedAssets(
            case_key=CaseKey(PAIR, FEED),
            restart_path=restart,
            template_deck_path=template,
            fallback_level=0,
            restart_provenance=f"pair_ecore:{RESTART}",
        ),
        {"i": 0},
    )


def _staged(tmp_path: Path, *, with_registry: bool):
    """Run one entry through a staging ``WaveVerifier``; return (outcome, decks)."""
    pkg = _package(tmp_path, with_registry=with_registry)
    template = tmp_path / "MAS_INP_cy12.inp"
    template.write_text(RELOAD_DECK, encoding="utf-8")
    restart = tmp_path / RESTART
    restart.write_bytes(b"restart")

    verifier = WaveVerifier(
        run_dir=tmp_path / "run",
        package_root=pkg,                 # NO resolver= — the defect's call shape
        evaluator_factory=lambda worker_id, cpu_core: StubEvaluator(),
        workers=1,
        stage_decks=True,
        library_dims=DIMS,
    )
    outcomes = verifier.evaluate_wave([_entry(tmp_path, template, restart)])
    decks = sorted((tmp_path / "run" / "produce_cases").glob("*/*/MAS_INP_cy*.inp"))
    return outcomes[0], decks


# --------------------------------------------------------------------------- #
# 1. the fixed emission path
# --------------------------------------------------------------------------- #
def test_verifier_emits_only_roster_aliases(tmp_path: Path) -> None:
    """A verifier pointed at a package whose ``registry.json`` carries the bridge
    stages a deck whose ``%LPD_SHF`` holds ONLY 2-char roster aliases."""
    outcome, decks = _staged(tmp_path, with_registry=True)

    assert outcome.status != "error", outcome.failure
    assert len(decks) == 1
    shf = extract_lpd_shf(decks[0].read_text(encoding="utf-8"))
    fresh = {tok.split()[1] for tok in shf.replace(",", "\n").splitlines()
             if tok.strip().upper().startswith("F ")}
    assert fresh, "no fresh cards in the staged SHF"
    assert fresh <= set(ALIASES), fresh
    for type_id in TYPE_IDS:
        assert type_id not in shf, f"raw type_id {type_id} reached the deck"


def test_registry_bridge_is_derived_from_the_package(tmp_path: Path) -> None:
    """``CaseAssetResolver(package_root)`` loads the package's own alias bridge —
    the fallback map was empty only because nobody read ``registry.json``."""
    pkg = _package(tmp_path, with_registry=True)
    assert registry_aliases_from_package(pkg) == REGISTRY
    assert CaseAssetResolver(pkg).type_to_alias == REGISTRY
    # ga80 (no registry.json) is unchanged: an empty bridge, i.e. a no-op.
    assert CaseAssetResolver(_package(tmp_path / "bare", with_registry=False)) \
        .type_to_alias == {}
    # An EXPLICIT empty map still means "no bridge" for a caller that means it.
    assert CaseAssetResolver(pkg, registry_aliases={}).type_to_alias == {}


# --------------------------------------------------------------------------- #
# 2. the guard itself
# --------------------------------------------------------------------------- #
def test_roster_parsed_from_lpd_bc() -> None:
    assert parse_lpd_bc_batch_ids(RELOAD_DECK) == ("R1", "R2", "S3", "S5", "P6")
    assert parse_lpd_bc_batch_ids("%LPD_SHF\n        F S3  0,\n") == ()


def test_guard_accepts_an_alias_only_deck() -> None:
    validate_reload_deck(RELOAD_DECK, RESTART, expected_dims=DIMS)  # no raise


def test_guard_rejects_a_raw_type_id_in_shf() -> None:
    """THE regression: the exact deck the defect emitted must not validate."""
    bad = RELOAD_DECK.replace("F S3  0,", f"F {TYPE_IDS[0]}  0,")
    with pytest.raises(DeckValidationError) as excinfo:
        validate_reload_deck(bad, RESTART, expected_dims=DIMS)
    message = str(excinfo.value)
    assert TYPE_IDS[0] in message
    assert "alias bridge" in message


def test_guard_rejects_an_off_roster_two_char_id() -> None:
    """A 2-char id MASTER would also absorb silently (not in ``%LPD_B&C``)."""
    bad = RELOAD_DECK.replace("F S3  0,", "F Z9  0,")
    with pytest.raises(DeckValidationError, match="LPD_B&C roster"):
        validate_reload_deck(bad, RESTART, expected_dims=DIMS)


def test_guard_honours_an_explicit_allowlist() -> None:
    validate_reload_deck(
        RELOAD_DECK, RESTART, expected_dims=DIMS, allowed_batch_ids=("S3", "S5")
    )
    with pytest.raises(DeckValidationError, match="LPD_B&C roster"):
        validate_reload_deck(
            RELOAD_DECK, RESTART, expected_dims=DIMS, allowed_batch_ids=("S5",)
        )


# --------------------------------------------------------------------------- #
# 3. the pre-fix construction can no longer emit silently
# --------------------------------------------------------------------------- #
def test_missing_bridge_fails_the_chain_instead_of_emitting(tmp_path: Path) -> None:
    """With the bridge unavailable (no ``registry.json``) the very first chain is
    an ``error`` with a deck-validation failure, and NO deck is left staged for
    MASTER — before the fix this staged a raw-``type_id`` deck and ran it."""
    outcome, decks = _staged(tmp_path, with_registry=False)

    assert outcome.status == "error"
    assert not decks, "a deck with untranslated type_ids must never be staged"
