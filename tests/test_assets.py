"""CaseAssetResolver — five-level fallback ladder, deck restart-reference rewrite
on a real GA candidate deck (byte-diff limited to SHF + restart), and atomic
promotion (plan section 5.2 / M2.5).
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from lpopt.search.assets import (
    CaseAssetResolver,
    DeckValidationError,
    _SYNTH_RESTART_PLACEHOLDER,
    _parse_lpd_cx_fuel_order,
    _read_deck_flex,
    synth_roster_for,
    validate_reload_deck,
)
from lpopt.design.coredeck import library_dims
from lpopt.search.genome import random_genome
from lpopt.vendor.masterrl.domain import CaseKey
from lpopt.vendor.masterrl.master import extract_lpd_shf

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_GLOB = (
    "3_GA_Surrogate/runs_flow/*/cases/*/*/candidates/*/rank_*_*/MAS_INP_cy*.inp"
)
NATIVE_F117_GLOB = "3_GA_Surrogate/FEASIBLE_PACKAGE/cores/K1_K2_f117/*/MAS_INP_cy*.inp"

# A minimal but structurally complete reload deck (restart-read equilibrium).
VALID_RELOAD_DECK = (
    "%JOB_TYP\n"
    "        1       stead                               # irrst=1 (restart)\n"
    "        MAS_RST.APRQ_11_0633.21\n"
    "        xsl     MAS_XSL\n"
    "        hff     MAS_HFF\n"
    "        out     MAS_OUT\n"
    "        sum     MAS_SUM\n"
    "%JOB_IDE\n"
    "        APRQ    12\n"
    "%GEN_DIM\n"
    "        10      10      27      83      85          # nx, ny, nz, nbatch, ncomp\n"
    "%LPD_SHF\n"
    "        F K1  0,\n"
    "%EXE_DEP                                            # BOC\n"
    "        0.0\n"
    "%EDT_OPT                                            # EOC restart write\n"
    "        1\n"
    "%END\n"
)

# A cycle-1-style fresh-core skeleton: batch map + no depletion chain.
SKELETON_DECK = (
    "%JOB_TYP\n"
    "        1       stead\n"
    "        MAS_RST.APRQ_11_0633.21\n"
    "%GEN_MTH\n"
    "        1\n"
    "%LPD_BCH\n"
    "        1 2 3\n"
    "%LPD_SHF\n"
    "        F K1  0,\n"
    "%END\n"
)


def _find_native_f117_deck() -> Path | None:
    for m in sorted(REPO_ROOT.parent.glob(NATIVE_F117_GLOB)):
        try:
            with open(m, "rb") as handle:
                handle.read(1)
            return m
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- #
# fake fuel library for level-3 (nearest e_core) resolution
# --------------------------------------------------------------------------- #
class _FakeFuel:
    """Minimal ``pair_e_core`` provider keyed by ``"a_b"`` pair string."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table

    def pair_e_core(self, a: str, b: str, split: float, library_id: str | None) -> float:
        key = f"{a}_{b}"
        if key not in self.table:
            raise KeyError(key)
        return self.table[key]


# --------------------------------------------------------------------------- #
# synthetic package tree
# --------------------------------------------------------------------------- #
def _make_restart(root: Path, folder: str, name: str) -> Path:
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"restart-" + name.encode())
    return p


def _make_deck(root: Path, folder: str, seed: str = "s1") -> Path:
    d = root / "cores" / folder / seed
    d.mkdir(parents=True, exist_ok=True)
    p = d / "MAS_INP_cy01.inp"
    p.write_text("dummy deck\n%LPD_SHF\n F K1  0,\n%END\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# fallback ladder
# --------------------------------------------------------------------------- #
def test_level0_native_exact(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2", "MAS_RST.NATIVE.01")
    _make_deck(pkg, "K1_K2")
    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted")

    res = r.resolve(CaseKey("K1_K2", 121))
    assert res.fallback_level == 0
    assert res.restart_provenance == "native:MAS_RST.NATIVE.01"
    assert res.template_deck_path is not None and res.template_deck_path.is_file()


def test_level1_promoted_exact(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2", "MAS_RST.NATIVE.01")  # only feed 121 native
    _make_deck(pkg, "K1_K2")
    promoted = tmp_path / "promoted"
    _make_restart(promoted, "K1_K2_f109", "MAS_RST.PROMOTED.09")

    r = CaseAssetResolver(pkg, promoted_root=promoted)
    res = r.resolve(("K1_K2", 109))
    assert res.fallback_level == 1
    assert res.restart_provenance == "promoted:MAS_RST.PROMOTED.09"


def test_level2_same_pair_nearest_feed(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2", "MAS_RST.F121.01")
    _make_restart(pkg / "bases", "K1_K2_f117", "MAS_RST.F117.01")
    _make_deck(pkg, "K1_K2")

    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted")
    # feed 105 has no exact folder -> nearest same-pair feed is 117.
    res = r.resolve(("K1_K2", 105))
    assert res.fallback_level == 2
    assert res.restart_provenance == "pair_feed:MAS_RST.F117.01"


def test_level3_nearest_ecore_pair(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2", "MAS_RST.K.01")
    _make_restart(pkg / "bases", "J1_J2", "MAS_RST.J.01")
    _make_deck(pkg, "K1_K2")
    fuel = _FakeFuel({"K1_K2": 5.20, "J1_J2": 5.10, "N1_N2": 5.40})

    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted", fuel_library=fuel)
    # N1_N2 has no folder and no same-pair fallback; nearest e_core is K1_K2
    # (|5.40-5.20|=0.20 < |5.40-5.10|=0.30).
    res = r.resolve(("N1_N2", 121))
    assert res.fallback_level == 3
    assert res.restart_provenance == "pair_ecore:MAS_RST.K.01"


def test_strict_restart_hard_errors_on_cross_pair_fallback(tmp_path: Path) -> None:
    # strict_restart=True: a level-3 cross-pair fallback (N1_N2 -> K1_K2 restart,
    # incompatible burnt types) must raise AssetResolutionError BEFORE Popen
    # instead of silently returning it (forensic 20260721 budget-burn fix).
    from lpopt.search.assets import AssetResolutionError
    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2", "MAS_RST.K.01")
    _make_deck(pkg, "K1_K2")
    fuel = _FakeFuel({"K1_K2": 5.20, "N1_N2": 5.40})

    strict = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted",
                               fuel_library=fuel, strict_restart=True)
    with pytest.raises(AssetResolutionError, match="no pair-matched restart"):
        strict.resolve(("N1_N2", 121))
    # a pair-matched (level 0) restart still resolves cleanly under strict mode.
    assert strict.resolve(("K1_K2", 121)).fallback_level == 0
    # default (strict off) still returns the graceful fallback (byte-identical).
    lax = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted", fuel_library=fuel)
    assert lax.resolve(("N1_N2", 121)).fallback_level == 3


def test_level4_neutral(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2", "MAS_RST.K.01")  # different pair only
    _make_deck(pkg, "K1_K2")
    neutral = tmp_path / "neutral" / "MAS_RST.NEUTRAL.00"
    neutral.parent.mkdir(parents=True)
    neutral.write_bytes(b"neutral")

    # No fuel library -> level 3 unavailable; unrelated pair -> level 4 neutral.
    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted", neutral_restart=neutral)
    res = r.resolve(("Z9_Z8", 121))
    assert res.fallback_level == 4
    assert res.restart_provenance == "neutral:MAS_RST.NEUTRAL.00"


def test_unresolved_restart(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _make_deck(pkg, "K1_K2")  # deck but no restarts anywhere, no neutral
    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted")
    res = r.resolve(("Z9_Z8", 121))
    assert res.fallback_level == -1
    assert res.restart_path is None
    assert res.restart_provenance == "unresolved:none"


# --------------------------------------------------------------------------- #
# per-library (paramA) alias bridge + deck-batch translation + dims attribute
# --------------------------------------------------------------------------- #
def test_registry_alias_bridge_ecore_resolves_alias_base_folder(tmp_path: Path) -> None:
    """A paramA band seed lives at bases/<ALIAS_pair>, but the produce case is a
    full ``type_id`` pair.  Without a registry the alias folder cannot be scored
    (level -1); with ``registry_aliases`` the e_core scorer bridges alias->type_id
    and resolves the band seed at level 3."""
    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "P0_P1", "MAS_RST.SEED.02")  # alias-named seed
    fuel = _FakeFuel({
        "TYPEA_TYPEB": 5.85,        # the produce case (type_id pair)
        "P5849X_P6257X": 5.68,      # the seed's translated type_id pair
    })
    reg = {"P5849X": "P0", "P6257X": "P1"}
    case = CaseKey("TYPEA_TYPEB", 117)

    # No registry -> the alias folder "P0_P1" cannot be scored -> unresolved.
    bare = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted", fuel_library=fuel)
    assert bare.resolve(case).fallback_level == -1

    # With the registry the alias folder is bridged to type_ids and scored.
    r = CaseAssetResolver(
        pkg, promoted_root=tmp_path / "promoted", fuel_library=fuel,
        registry_aliases=reg, library_id="paramA",
    )
    res = r.resolve(case)
    assert res.fallback_level == 3
    assert res.restart_provenance == "pair_ecore:MAS_RST.SEED.02"


def test_prepare_cycle1_deck_translates_type_id_batches_to_aliases() -> None:
    """For a paramA package the pattern carries full ``type_id`` fresh-batch
    labels, but the reload deck's ``%LPD_B&C`` ids are the 2-char aliases; the
    prepared SHF must reference the aliases so MASTER binds the loading."""
    reg = {"P6253Z1G08N20": "P2", "P6257Z1G06N24": "P6"}
    r = CaseAssetResolver(".", registry_aliases=reg, library_id="paramA")
    pair = "P6253Z1G08N20_P6257Z1G06N24"
    pattern = random_genome(random.Random(3), pair, 30).to_pattern()
    # sanity: the pattern itself is in type_id space
    assert {i.batch for i in pattern.items if i.is_fresh} == set(pair.split("_"))

    prepared = r.prepare_cycle1_deck(VALID_RELOAD_DECK, pattern, "MAS_RST.APRQ_11_0633.21")
    shf = extract_lpd_shf(prepared)
    fresh_ids = set(re.findall(r"F\s+([A-Za-z0-9]+)\s", shf))
    assert fresh_ids == {"P2", "P6"}, fresh_ids
    assert "P6253Z1G08N20" not in shf and "P6257Z1G06N24" not in shf


def test_prepare_cycle1_deck_ga80_no_alias_translation() -> None:
    """ga80 (no registry) is a byte-for-byte no-op: the pattern's own labels flow
    into the SHF unchanged."""
    r = CaseAssetResolver(".")  # no registry_aliases
    pattern = random_genome(random.Random(4), "K1_K2", 30).to_pattern()
    prepared = r.prepare_cycle1_deck(VALID_RELOAD_DECK, pattern, "MAS_RST.APRQ_11_0633.21")
    got = extract_lpd_shf(prepared).replace("\r\n", "\n").strip()
    want = pattern.to_shf().replace("\r\n", "\n").strip()
    assert got == want


def test_library_dims_attribute_default_and_custom() -> None:
    from lpopt.search.assets import LIBRARY_DIMS

    assert CaseAssetResolver(".").library_dims == LIBRARY_DIMS
    assert CaseAssetResolver(".", library_dims=(14, 16)).library_dims == (14, 16)


def test_alias_case_key_and_pattern_match_for_runner() -> None:
    """The vendor runner re-injects ``pattern.to_shf()`` and validates
    ``pattern.validate_case(key.pair, feed)`` every cycle, so the alias-translated
    case key and pattern must agree — a full type_id pair maps to the alias pair,
    and the alias pattern validates against it (ga80 is a no-op)."""
    reg = {"P6253Z1G08N20": "P2", "P6257Z1G06N24": "P6"}
    r = CaseAssetResolver(".", registry_aliases=reg, library_id="paramA")
    pair = "P6253Z1G08N20_P6257Z1G06N24"
    ck = CaseKey(pair, 121)
    run_key = r.alias_case_key(ck)
    assert run_key.pair == "P2_P6" and run_key.feed == 121

    pat = random_genome(random.Random(9), pair, 30).to_pattern()
    run_pat = r.alias_pattern(pat)
    # the alias pattern validates against the alias key (the runner's contract)
    run_pat.validate_case(run_key.pair, run_key.feed)
    assert {i.batch for i in run_pat.items if i.is_fresh} == {"P2", "P6"}

    # ga80 (no registry) — both are identity no-ops
    g = CaseAssetResolver(".")
    gk = CaseKey("K1_K2", 121)
    assert g.alias_case_key(gk) == gk
    gpat = random_genome(random.Random(9), "K1_K2", 30).to_pattern()
    assert g.alias_pattern(gpat) is gpat


# --------------------------------------------------------------------------- #
# deck rewrite on a REAL candidate deck
# --------------------------------------------------------------------------- #
def _find_candidate_deck() -> Path | None:
    matches = sorted(REPO_ROOT.parent.glob(CANDIDATE_GLOB))
    for m in matches:
        try:
            with open(m, "rb") as handle:
                handle.read(1)
            return m
        except OSError:
            continue
    return None


def test_prepare_cycle1_deck_byte_scoped() -> None:
    deck_path = _find_candidate_deck()
    if deck_path is None:
        pytest.skip("no readable runs_flow candidate deck available")

    original = _read_deck_flex(deck_path)
    resolver = CaseAssetResolver(".")

    # A different pattern (feed 121) so the SHF body genuinely changes.
    pattern = random_genome(random.Random(20), "K1_K2", 30).to_pattern()
    fake_restart = "MAS_RST.FAKE_99_0001.23"
    new = resolver.prepare_cycle1_deck(original, pattern, fake_restart)

    old_lines = original.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    assert len(old_lines) == len(new_lines)
    changed = [i for i in range(len(old_lines)) if old_lines[i] != new_lines[i]]

    # Locate the %LPD_SHF body span and the single restart-reference line.
    shf_start = next(
        i for i, ln in enumerate(old_lines) if ln.lstrip().upper().startswith("%LPD_SHF")
    )
    shf_end = next(
        i for i in range(shf_start + 1, len(old_lines)) if old_lines[i].lstrip().startswith("%")
    )
    shf_body = set(range(shf_start + 1, shf_end))
    restart_line = next(
        i for i, ln in enumerate(old_lines) if "MAS_RST.APRQ_11_0652.86" in ln
    )
    allowed = shf_body | {restart_line}

    assert set(changed) <= allowed, f"unexpected changes outside SHF/restart: {set(changed)-allowed}"
    assert restart_line in changed
    # The %LPD_SHF card body now holds exactly the new pattern's shuffle cards
    # (replace_lpd_shf normalizes the body to the deck's own \r\n line ending).
    got = extract_lpd_shf(new).replace("\r\n", "\n").strip()
    want = pattern.to_shf().replace("\r\n", "\n").strip()
    assert got == want
    # Old restart fully gone; new restart present.
    assert "MAS_RST.APRQ_11_0652.86" not in new
    assert fake_restart in new_lines[restart_line]


# --------------------------------------------------------------------------- #
# promotion
# --------------------------------------------------------------------------- #
def test_promote_atomic_single_restart(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    promoted = tmp_path / "promoted"
    r = CaseAssetResolver(pkg, promoted_root=promoted)

    src = tmp_path / "MAS_RST.SOURCE.11"
    src.write_bytes(b"restart-bytes-123")
    dest = r.promote(("K1_K2", 109), src)

    assert dest.exists()
    assert dest.read_bytes() == b"restart-bytes-123"
    assert dest.parent.name == "K1_K2_f109"
    assert list(dest.parent.glob("MAS_RST.*")) == [dest]
    assert not list(dest.parent.glob(".*tmp*")), "temp file left behind (not atomic)"

    # A later, differently-named promotion replaces the folder's single restart.
    src2 = tmp_path / "MAS_RST.SOURCE.22"
    src2.write_bytes(b"newer-restart")
    dest2 = r.promote(("K1_K2", 109), src2)
    assert list(dest2.parent.glob("MAS_RST.*")) == [dest2]
    assert dest2.read_bytes() == b"newer-restart"

    # Promotion then resolves as level 1 (promoted exact).
    _make_restart(pkg / "bases", "OTHER", "MAS_RST.X")  # unrelated native
    res = r.resolve(("K1_K2", 109))
    assert res.fallback_level == 1
    assert res.restart_provenance == "promoted:MAS_RST.SOURCE.22"


# --------------------------------------------------------------------------- #
# template resolution prefers a RELOAD deck over a cy1 skeleton
# --------------------------------------------------------------------------- #
def test_resolve_prefers_reload_deck_over_skeleton(tmp_path: Path) -> None:
    """A cy1 skeleton that sorts first must not win over a real reload deck."""

    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2_f117", "MAS_RST.APRQ_11_0633.21")

    core = pkg / "cores" / "K1_K2_f117"
    # 'aaa' sorts before 'zzz' -> the skeleton is the first glob hit.
    skel = core / "aaa_skeleton"
    skel.mkdir(parents=True)
    (skel / "MAS_INP_cy01.inp").write_text(SKELETON_DECK, encoding="utf-8")
    good = core / "zzz_reload"
    good.mkdir(parents=True)
    (good / "MAS_INP_cy12.inp").write_text(VALID_RELOAD_DECK, encoding="utf-8")

    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted")
    res = r.resolve(CaseKey("K1_K2", 117))
    assert res.fallback_level == 0
    assert res.template_deck_path == good / "MAS_INP_cy12.inp"
    assert not res.deck_fallback  # the exact reload deck, not a fallback


def test_resolve_falls_back_when_no_reload_deck(tmp_path: Path) -> None:
    """With only a non-reload deck present, resolution still returns it (pass 2)."""

    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2_f117", "MAS_RST.X")
    only = pkg / "cores" / "K1_K2_f117" / "s1"
    only.mkdir(parents=True)
    (only / "MAS_INP_cy01.inp").write_text(SKELETON_DECK, encoding="utf-8")

    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted")
    res = r.resolve(CaseKey("K1_K2", 117))
    assert res.template_deck_path == only / "MAS_INP_cy01.inp"


# --------------------------------------------------------------------------- #
# deck sanity gate (validate_reload_deck)
# --------------------------------------------------------------------------- #
RESTART = "MAS_RST.APRQ_11_0633.21"


def test_validate_reload_deck_accepts_valid() -> None:
    validate_reload_deck(VALID_RELOAD_DECK, RESTART)  # does not raise


def test_validate_reload_deck_rejects_missing_exe_dep() -> None:
    bad = VALID_RELOAD_DECK.replace("%EXE_DEP                                            # BOC\n        0.0\n", "")
    with pytest.raises(DeckValidationError, match="EXE_DEP"):
        validate_reload_deck(bad, RESTART)


def test_validate_reload_deck_rejects_missing_edt_opt() -> None:
    bad = VALID_RELOAD_DECK.replace("%EDT_OPT                                            # EOC restart write\n        1\n", "")
    with pytest.raises(DeckValidationError, match="EDT_OPT"):
        validate_reload_deck(bad, RESTART)


def test_validate_reload_deck_rejects_lpd_bch() -> None:
    bad = VALID_RELOAD_DECK.replace("%LPD_SHF\n", "%LPD_BCH\n        1 2 3\n%LPD_SHF\n")
    with pytest.raises(DeckValidationError, match="LPD_BCH"):
        validate_reload_deck(bad, RESTART)


def test_validate_reload_deck_rejects_wrong_restart() -> None:
    with pytest.raises(DeckValidationError, match="restart reference"):
        validate_reload_deck(VALID_RELOAD_DECK, "MAS_RST.OTHER.99")


def test_validate_reload_deck_rejects_wrong_dims() -> None:
    with pytest.raises(DeckValidationError, match="GEN_DIM"):
        validate_reload_deck(VALID_RELOAD_DECK, RESTART, expected_dims=(80, 80))


# --------------------------------------------------------------------------- #
# prepare_cycle1_deck byte-scoping + validation on the REAL f117 native template
# --------------------------------------------------------------------------- #
def test_prepare_cycle1_deck_byte_scoped_f117_native() -> None:
    deck_path = _find_native_f117_deck()
    if deck_path is None:
        pytest.skip("no readable native f117 template available")

    original = _read_deck_flex(deck_path)
    resolver = CaseAssetResolver(".")
    pattern = random_genome(random.Random(117), "K1_K2", 29).to_pattern()  # feed 117
    assert pattern.feed == 117
    new_restart = "MAS_RST.FAKE_99_0001.23"
    prepared = resolver.prepare_cycle1_deck(original, pattern, new_restart)

    old_lines = original.splitlines(keepends=True)
    new_lines = prepared.splitlines(keepends=True)
    assert len(old_lines) == len(new_lines)
    changed = [i for i in range(len(old_lines)) if old_lines[i] != new_lines[i]]

    shf_start = next(
        i for i, ln in enumerate(old_lines) if ln.lstrip().upper().startswith("%LPD_SHF")
    )
    shf_end = next(
        i for i in range(shf_start + 1, len(old_lines)) if old_lines[i].lstrip().startswith("%")
    )
    shf_body = set(range(shf_start + 1, shf_end))
    restart_line = next(
        i for i, ln in enumerate(old_lines) if "MAS_RST.APRQ_11_0633.21" in ln
    )
    allowed = shf_body | {restart_line}
    assert set(changed) <= allowed, f"unexpected changes: {set(changed) - allowed}"
    assert restart_line in changed

    got = extract_lpd_shf(prepared).replace("\r\n", "\n").strip()
    want = pattern.to_shf().replace("\r\n", "\n").strip()
    assert got == want
    assert "MAS_RST.APRQ_11_0633.21" not in prepared
    assert new_restart in new_lines[restart_line]

    # The prepared native reload deck passes the pre-Popen sanity gate.
    validate_reload_deck(prepared, new_restart)


# --------------------------------------------------------------------------- #
# cross-family reload-template synthesis (plan 5.2 / 12.1)
# --------------------------------------------------------------------------- #
FEASIBLE_PACKAGE = REPO_ROOT.parent / "3_GA_Surrogate" / "FEASIBLE_PACKAGE"


def _first_readable_vendor_deck() -> Path | None:
    cores = FEASIBLE_PACKAGE / "cores"
    if not cores.is_dir():
        return None
    for m in sorted(cores.glob("*/*/MAS_INP_cy*.inp")):
        try:
            with open(m, "rb") as handle:
                handle.read(1)
            return m
        except OSError:
            continue
    return None


def _synth_resolver(tmp_path: Path, package_root: Path | None = None, **kw) -> CaseAssetResolver:
    pkg = package_root if package_root is not None else (tmp_path / "pkg")
    (pkg / "bases").mkdir(parents=True, exist_ok=True)
    return CaseAssetResolver(
        pkg,
        promoted_root=tmp_path / "promoted",
        synth_root=tmp_path / "synth_decks",
        library_id="ga80",
        **kw,
    )


def test_synth_roster_for_ga80() -> None:
    roster = synth_roster_for("ga80")
    assert len(roster) == 80
    # the cross-family batches that motivate synthesis are all present
    for batch in ("J2", "L3", "J4", "K6", "F1", "M1", "N6"):
        assert batch in roster
    assert library_dims(len(roster)) == (83, 85)
    assert synth_roster_for("unknown_lib") == ()


def test_synth_cross_family_j2_l3(tmp_path: Path) -> None:
    """A cross-family pair with no shipped deck now resolves to a synthesized
    reload template (was MissingCaseAssetError)."""

    r = _synth_resolver(tmp_path)  # empty package -> GA80 roster fallback
    res = r.resolve(CaseKey("J2_L3", 121))

    assert res.template_deck_path is not None and res.template_deck_path.is_file()
    assert res.template_deck_path == tmp_path / "synth_decks" / "J2_L3" / "MAS_INP_cy12.inp"
    assert any("synthesized" in n for n in res.notes), res.notes

    text = _read_deck_flex(res.template_deck_path)
    # byte-compatible: passes the pre-Popen reload-deck sanity gate for the library
    validate_reload_deck(text, _SYNTH_RESTART_PLACEHOLDER, expected_dims=(83, 85))
    # every roster type (incl. the pair's J2/L3) is a defined batch
    order = _parse_lpd_cx_fuel_order(text)
    assert set(order) == set(synth_roster_for("ga80"))


def test_synth_byte_compat_vendor_round_trip(tmp_path: Path) -> None:
    """The synthesized deck survives the exact vendor primitives the harness runs:
    replace_lpd_shf + advance_cycle_deck (via prepare_cycle1_deck), then the
    pre-Popen validate_reload_deck against the real staged restart."""

    r = _synth_resolver(tmp_path)
    res = r.resolve(CaseKey("J2_L3", 121))
    template = _read_deck_flex(res.template_deck_path)

    pattern = random_genome(random.Random(23), "J2_L3", 30).to_pattern()
    real_restart = "MAS_RST.APRQ_11_0652.86"
    prepared = r.prepare_cycle1_deck(template, pattern, real_restart)

    # passes the sanity gate against the now-staged restart (dims = library)
    validate_reload_deck(prepared, real_restart, expected_dims=(83, 85))
    # the SHF body is exactly the pattern's shuffle (byte round-trip modulo eol)
    got = extract_lpd_shf(prepared).replace("\r\n", "\n").strip()
    want = pattern.to_shf().replace("\r\n", "\n").strip()
    assert got == want
    # the restart reference was rewritten from the placeholder to the real restart
    assert _SYNTH_RESTART_PLACEHOLDER not in prepared
    assert real_restart in prepared


def test_synth_prefers_packaged_deck(tmp_path: Path) -> None:
    """A packaged reload deck wins over synthesis when one exists for the pair."""

    pkg = tmp_path / "pkg"
    _make_restart(pkg / "bases", "K1_K2", "MAS_RST.NATIVE.01")
    deck_dir = pkg / "cores" / "K1_K2" / "s1"
    deck_dir.mkdir(parents=True)
    (deck_dir / "MAS_INP_cy12.inp").write_text(VALID_RELOAD_DECK, encoding="utf-8")

    r = _synth_resolver(tmp_path, package_root=pkg)
    res = r.resolve(CaseKey("K1_K2", 121))

    assert res.template_deck_path == deck_dir / "MAS_INP_cy12.inp"
    assert not any("synthesized" in n for n in res.notes), res.notes
    # nothing was written to the synth cache
    assert not (tmp_path / "synth_decks" / "K1_K2").exists()


def test_synth_cache_reuse(tmp_path: Path) -> None:
    """A synthesized deck is cached and reused verbatim on the next resolve."""

    r = _synth_resolver(tmp_path)
    res1 = r.resolve(CaseKey("J4_K6", 121))
    assert any(n.startswith("deck: synthesized (J4_K6") for n in res1.notes), res1.notes
    first_bytes = res1.template_deck_path.read_bytes()
    first_mtime = res1.template_deck_path.stat().st_mtime_ns

    res2 = r.resolve(CaseKey("J4_K6", 121))
    assert res2.template_deck_path == res1.template_deck_path
    assert any("cache reuse" in n for n in res2.notes), res2.notes
    # reused verbatim: identical bytes, not rewritten
    assert res2.template_deck_path.read_bytes() == first_bytes
    assert res2.template_deck_path.stat().st_mtime_ns == first_mtime


def test_synth_disabled_without_root(tmp_path: Path) -> None:
    """Without synth_root the legacy behavior is preserved: no template, no cache."""

    pkg = tmp_path / "pkg"
    (pkg / "bases").mkdir(parents=True)
    r = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted", library_id="ga80")
    res = r.resolve(CaseKey("J2_L3", 121))
    assert res.template_deck_path is None
    assert any(n.startswith("deck: NONE") for n in res.notes)


def test_synth_explicit_roster_overrides_and_sets_dims(tmp_path: Path) -> None:
    """An explicit roster is used verbatim and drives the GEN_DIM library dims."""

    roster = ["J2", "L3", "K1"]
    r = _synth_resolver(tmp_path, synth_roster=roster)
    res = r.resolve(CaseKey("J2_L3", 121))
    text = _read_deck_flex(res.template_deck_path)

    dims = library_dims(len(roster))  # (3+3, 5+3) = (6, 8)
    assert dims == (6, 8)
    validate_reload_deck(text, _SYNTH_RESTART_PLACEHOLDER, expected_dims=dims)
    assert list(_parse_lpd_cx_fuel_order(text)) == roster


def test_vendor_fuel_order_prefers_dim_consistent_deck(tmp_path: Path) -> None:
    """Root-cause regression (paramA 6.x-band blocker): a STALE cores bootstrap
    deck (built when the library had fewer fuel types) that sorts alphabetically
    FIRST must not poison the synthesis roster.  ``_vendor_fuel_order`` prefers
    the first deck whose roster is dimension-consistent with the library."""

    from lpopt.design.coredeck import build_reload_deck, library_dims

    pkg = tmp_path / "pkg"
    stale = [f"T{i:02d}" for i in range(11)]     # -> library_dims(11) = (14,16)
    current = [f"U{i:02d}" for i in range(33)]   # -> library_dims(33) = (36,38)
    # "AA_stale" sorts before "ZZ_curr": the stale deck wins under naive `sorted`.
    for name, roster in (("AA_stale", stale), ("ZZ_curr", current)):
        d = pkg / "cores" / name / "bootstrap"
        d.mkdir(parents=True)
        (d / "MAS_INP_cy02.inp").write_text(
            build_reload_deck(roster, _SYNTH_RESTART_PLACEHOLDER, 2), encoding="utf-8"
        )
    (pkg / "bases").mkdir(parents=True, exist_ok=True)

    r = CaseAssetResolver(
        pkg, promoted_root=tmp_path / "promoted",
        library_id="paramA", library_dims=library_dims(33),
    )
    # the (36,38)-consistent deck is chosen despite the stale one sorting first
    assert list(r._vendor_fuel_order(pkg)) == current
    assert library_dims(len(r._vendor_fuel_order(pkg))) == (36, 38)


def test_reload_gate_message_suggests_rebuild_on_dim_mismatch() -> None:
    """The reload-deck sanity gate's %GEN_DIM-mismatch error names the fix
    (rebuild the band seeds via `lpopt design bootstrap`) so the failure is
    actionable, not cryptic."""

    from lpopt.design.coredeck import build_reload_deck, library_dims

    deck = build_reload_deck([f"T{i:02d}" for i in range(11)], _SYNTH_RESTART_PLACEHOLDER, 2)
    with pytest.raises(DeckValidationError) as exc:
        validate_reload_deck(deck, _SYNTH_RESTART_PLACEHOLDER, expected_dims=library_dims(33))
    msg = str(exc.value)
    assert "!= library" in msg
    assert "(14, 16)" in msg and "(36, 38)" in msg
    assert "design bootstrap" in msg and "--feed 121" in msg


def test_synth_cx_order_matches_vendor(tmp_path: Path) -> None:
    """When the package ships a deck, synthesis reuses the vendor's exact
    composition-index -> FA-name binding (so it matches the base restart)."""

    vendor_deck = _first_readable_vendor_deck()
    if vendor_deck is None:
        pytest.skip("no readable FEASIBLE_PACKAGE vendor deck available")

    vendor_order = _parse_lpd_cx_fuel_order(_read_deck_flex(vendor_deck))
    assert len(vendor_order) == 80

    r = _synth_resolver(tmp_path, package_root=FEASIBLE_PACKAGE)
    res = r.resolve(CaseKey("J2_L3", 121))
    assert any("synthesized" in n for n in res.notes), res.notes
    synth_order = _parse_lpd_cx_fuel_order(_read_deck_flex(res.template_deck_path))

    # identical index -> name binding as the decks the base restarts were written under
    assert synth_order == vendor_order


# --------------------------------------------------------------------------- #
# fr_arms.py — the two silent failures around a FIXED-pattern fuel arm
# --------------------------------------------------------------------------- #
def _fr_arms_module():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import fr_arms                                        # noqa: PLC0415

    return fr_arms


def _fr_arms_fixture(tmp_path: Path, monkeypatch, fallback_level: int):
    """Wire fr_arms.main() to a fake package + fake resolver/verifier."""
    import numpy as np

    from lpopt.search import assets as A
    from lpopt.search import verify as V
    from lpopt.search.assets import ResolvedAssets

    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True, exist_ok=True)
    (pkg / "lib" / "MAS_XSL").write_text(
        "COMP FA_E1 x\nCOMP FA_E2 x\nREFL R1 x\n", encoding="utf-8")

    class _FakeResolver:
        def __init__(self, *a, **kw):
            pass

        def resolve(self, key):
            return ResolvedAssets(case_key=key, restart_path=None,
                                  template_deck_path=None,
                                  fallback_level=fallback_level,
                                  restart_provenance="fake:MAS_RST")

    class _Outcome:
        status, n_cycles, wall_s = "converged", 11, 1.0
        restart_provenance, failure, fom = "fake:MAS_RST", "", None
        maps = np.ones((4, 9, 9), dtype=float)

        def __init__(self, entry):
            self.case_key, self.meta = entry.case_key, entry.meta

    class _FakeVerifier:
        def __init__(self, *a, **kw):
            pass

        def evaluate_wave(self, entries):
            return [_Outcome(e) for e in entries]

    monkeypatch.setattr(A, "CaseAssetResolver", _FakeResolver)
    monkeypatch.setattr(V, "WaveVerifier", _FakeVerifier)
    return pkg


def test_fr_arms_skips_an_arm_that_resolves_at_a_fallback_restart(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """A fallback restart is a CONFOUND for the fuel measurement.

    Every arm shares one pattern and one feed so that only the fresh batch
    identities differ; a restart that is not the pair's own reintroduces exactly
    the variable the design removed.  fr_arms printed the level and ran anyway
    (ECC audit) — it now banners and skips unless --allow-fallback.
    """
    fr_arms = _fr_arms_module()
    pkg = _fr_arms_fixture(tmp_path, monkeypatch, fallback_level=2)
    argv = ["fr_arms.py", "--arm", "A0", "--package", str(pkg),
            "--run-dir", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc:
        fr_arms.main()
    assert "no runnable arms" in str(exc.value)
    out = capsys.readouterr().out
    assert "fallback_level=2" in out and "A0 SKIPPED" in out

    monkeypatch.setattr("sys.argv", argv + ["--allow-fallback"])
    fr_arms.main()                                        # opt-in still runs
    assert "RUN ANYWAY" in capsys.readouterr().out


def test_fr_arms_never_overwrites_a_previously_saved_map(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """A re-run of an arm must not replace the plane a published number came from.

    The verifier purges the case outputs after harvest, so an overwritten map is
    an unrepeatable measurement.  Primary naming (``map_<arm>.npy``, what the kit
    builders read) is deliberately unchanged (ECC audit).
    """
    fr_arms = _fr_arms_module()
    pkg = _fr_arms_fixture(tmp_path, monkeypatch, fallback_level=0)
    run_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["fr_arms.py", "--arm", "A0", "--package",
                                     str(pkg), "--run-dir", str(run_dir)])
    fr_arms.main()
    primary = run_dir / "map_A0.npy"
    assert primary.is_file()
    original = primary.read_bytes()

    capsys.readouterr()
    fr_arms.main()
    out = capsys.readouterr().out
    assert (run_dir / "map_A0_1.npy").is_file()
    assert primary.read_bytes() == original           # the first plane survives
    assert "WARNING" in out and "map_A0_1.npy" in out


def test_fr_arms_invariants_fire_when_a_substitution_moves_the_feed(
        tmp_path: Path, monkeypatch) -> None:
    """The arms are comparable only because ONLY batch NAMES change.

    A substitution that moved the feed (or the size of the batch multiset) would
    make the arm a different core, not a different fuel — so it is a hard exit,
    checked before ~40 MASTER-minutes are spent.
    """
    fr_arms = _fr_arms_module()
    pkg = _fr_arms_fixture(tmp_path, monkeypatch, fallback_level=0)
    monkeypatch.setattr("sys.argv", ["fr_arms.py", "--arm", "A0", "--package",
                                     str(pkg), "--run-dir", str(tmp_path / "out")])

    real = fr_arms.substitute

    def _drops_a_fresh_card(pattern, mapping):
        out = real(pattern, mapping)
        items = list(out.items)
        spare = next(it for it in items if not it.is_fresh)
        for i, it in enumerate(items):          # turn the first fresh card burnt
            if it.is_fresh:
                items[i] = spare
                break
        return type(pattern)(tuple(items))

    monkeypatch.setattr(fr_arms, "substitute", _drops_a_fresh_card)
    with pytest.raises(SystemExit) as exc:
        fr_arms.main()
    assert "batch-multiset size" in str(exc.value)
