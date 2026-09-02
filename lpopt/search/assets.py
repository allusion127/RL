"""Case asset resolution for the production campaign (plan sections 5.2 / M2.5).

A new ``(pair, feed)`` DoE cell rarely has its own MASTER base restart or its own
template deck.  :class:`CaseAssetResolver` finds the best available *restart* and
a readable *template deck* for any case, using a five-level fallback ladder, and
rewrites the deck's restart reference so a fallback restart is accepted by the
strict :class:`~lpopt.vendor.masterrl.master.MasterRunner`.

Restart fallback ladder (``fallback_level`` 0..4)::

    0  native      package_root/bases/<folder>/MAS_RST.*        (exact case)
    1  promoted     promoted_root/<folder>/MAS_RST.*            (self-improving cache)
    2  pair_feed    same pair, nearest feed folder
    3  pair_ecore   nearest e_core pair (via FuelLibrary), same-or-nearest feed
    4  neutral      configured neutral warm restart

Template deck resolution (independent of the restart level)::

    exact case cores/<folder>/*/MAS_INP_cy*.inp
    -> any same-pair deck
    -> any readable deck matched by the configured ``template_fallbacks`` globs
    -> SYNTHESIZED reload template (coredeck) for the library's full roster

The final tier closes the cross-family gap: the FEASIBLE_PACKAGE ships template
decks only for same-family pairs, so a cross-family case (e.g. ``J2_L3``) has no
exact/same-pair deck and previously failed with ``MissingCaseAssetError``.  When
synthesis is enabled (``synth_root`` set + a known/explicit roster),
:meth:`CaseAssetResolver._synthesize_template` builds a restart-read reload deck
from the library's composition tables via
:func:`lpopt.design.coredeck.build_reload_deck` — the same validated primitive
the Phase-A chain uses — and caches it under ``synth_root/<pair>/`` for reuse
across campaigns.  The roster order is taken from a shipped vendor deck's
``%LPD_C&X`` when available, so the synthesized composition-index -> ``FA_<name>``
binding matches the decks the base restarts were written under.  The synthesized
deck is byte-compatible with what the vendor harness parses: it passes
:func:`validate_reload_deck` and survives the ``replace_lpd_shf`` +
``advance_cycle_deck`` round-trip that :meth:`prepare_cycle1_deck` performs.

Every resolution carries a ``restart_provenance`` string
(``"<kind>:<basename>"``) that flows verbatim into the store's
``restart_provenance`` column: non-convergence is a property of the
``(pattern, restart, max_cycles)`` triple, so the store must remember which
restart produced each label (plan 5.4).

The deck rewrite mirrors the vendor
:func:`~lpopt.vendor.masterrl.equilibrium.advance_cycle_deck` mechanism: that
function is the single audited place that rewrites the sole ``%JOB_TYP`` restart
reference (and keeps ``%JOB_IDE`` / ``%JOB_TIT`` consistent).  ``prepare_cycle1_deck``
reuses it verbatim with the deck's own cycle number so *only* the restart
reference and the ``%LPD_SHF`` body change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from ..data.fuel_types import case_e_core as _case_e_core
from .genome import MAX_FRESH_TYPES
from ..vendor.masterrl.domain import CaseKey, Pattern
from ..vendor.masterrl.equilibrium import advance_cycle_deck, deck_cycle
from ..vendor.masterrl.master import extract_lpd_shf, replace_lpd_shf


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #
_LEVEL_KIND = {
    0: "native",
    1: "promoted",
    2: "pair_feed",
    3: "pair_ecore",
    4: "neutral",
}


@dataclass(frozen=True)
class ResolvedAssets:
    """The restart + template deck resolved for one produce case."""

    case_key: CaseKey
    restart_path: Path | None
    template_deck_path: Path | None
    fallback_level: int          # 0..4 (restart ladder); -1 when nothing found
    restart_provenance: str      # "<kind>:<basename>" (flows into store rows)
    notes: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return _LEVEL_KIND.get(self.fallback_level, "unresolved")

    @property
    def deck_fallback(self) -> bool:
        """True when the template deck is not the exact case's own deck."""

        return any(note.startswith("deck:") and "exact" not in note for note in self.notes)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_case_key(case_key: CaseKey | tuple[str, int]) -> CaseKey:
    if isinstance(case_key, CaseKey):
        return case_key
    pair, feed = case_key
    return CaseKey(str(pair), int(feed))


def _feed_of_folder(folder: str, pair: str) -> int | None:
    """Recover the feed a bases/cores folder encodes (``<pair>`` or ``<pair>_f<n>``)."""

    if folder == pair:
        return 121
    prefix = f"{pair}_f"
    if folder.startswith(prefix):
        tail = folder[len(prefix):]
        if tail.isdigit():
            return int(tail)
    return None


def _pair_of_folder(folder: str) -> str:
    """Strip a trailing ``_f<feed>`` suffix to recover the pair name."""

    marker = folder.rfind("_f")
    if marker > 0 and folder[marker + 2:].isdigit():
        return folder[:marker]
    return folder


def _only_restart(directory: Path) -> Path | None:
    """The single ``MAS_RST.*`` in a directory (or ``None``); prefers readable."""

    if not directory.is_dir():
        return None
    candidates = sorted(p for p in directory.glob("MAS_RST.*") if p.is_file())
    if not candidates:
        return None
    for candidate in candidates:
        if _is_readable(candidate):
            return candidate
    return candidates[0]


def _is_readable(path: Path) -> bool:
    """Open-and-read one byte — guards against OneDrive dehydrated placeholders."""

    try:
        with open(path, "rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _read_deck_flex(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decoding is total")


@dataclass(frozen=True)
class _RestartEntry:
    pair: str
    feed: int
    path: Path
    source: str  # "native" | "promoted"


# --------------------------------------------------------------------------- #
# reload-deck detection + pre-execution sanity gate
# --------------------------------------------------------------------------- #
class AssetResolutionError(RuntimeError):
    """A restart resolved only to an INCOMPATIBLE cross-pair fallback.

    Raised (opt-in, ``strict_restart=True``) when the best restart is a level-3
    nearest-e_core / level-4 neutral candidate from a DIFFERENT pair than the
    requested case.  Such a restart carries burnt-assembly types that are absent
    from the target ga80 deck's ``%LPD_B&C``, so MASTER dies at INITIALIZE
    ("MAS_SUM EDIT 2 header anchor was not found") after emitting
    "##### ***NOT DEFINED IN LPD_B&C" — a HARD configuration error that must be
    surfaced BEFORE Popen, never a per-call soft error that silently burns budget
    (forensic 20260721: the min_fuel_cost outer search proposed cells whose pair
    had no native restart).  The fix is a pair-matched (level 0-2) restart or
    restricting the search to restart-bearing pairs.
    """


class DeckValidationError(ValueError):
    """A prepared reload deck failed the pre-execution (pre-Popen) sanity gate.

    A restart-read steady-state deck that is missing its depletion chain
    (``%EXE_DEP``) / EOC restart write (``%EDT_OPT``), or that still carries a
    fresh-core batch map (``%LPD_BCH``), or whose ``%JOB_TYP`` restart reference
    does not match the staged ``MAS_RST.*`` will drive MASTER into a non-finite
    (NaN) multigroup-outer loop.  This gate refuses such a deck *before* MASTER
    is ever launched (plan 5.2 / M2.5).
    """


#: MASTER library dimensions for the ga80 FEASIBLE_PACKAGE (nbatch, ncomp).
LIBRARY_DIMS = (83, 85)

_DECK_CARD_RE = re.compile(r"^[ \t]*%(?P<name>[A-Za-z0-9_]+)", re.MULTILINE)


def _card_count(deck: str, card: str) -> int:
    """Number of ``%<card>`` headers in ``deck`` (case-insensitive)."""

    target = card.upper()
    return sum(1 for m in _DECK_CARD_RE.finditer(deck) if m.group("name").upper() == target)


def _card_data_lines(deck: str, card: str) -> list[str]:
    """Non-comment, non-blank data lines belonging to the first ``%<card>``."""

    lines = deck.splitlines()
    target = card.upper()
    starts = [
        i
        for i, line in enumerate(lines)
        if (m := _DECK_CARD_RE.match(line)) is not None and m.group("name").upper() == target
    ]
    if not starts:
        return []
    start = starts[0]
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _DECK_CARD_RE.match(lines[j]) is not None:
            end = j
            break
    out: list[str] = []
    for j in range(start + 1, end):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("#") or stripped == "/":
            continue
        out.append(lines[j])
    return out


def _is_reload_deck(deck: str) -> bool:
    """A restart-read equilibrium deck: has a depletion chain + shuffle, no
    fresh-core batch map."""

    return (
        _card_count(deck, "EXE_DEP") >= 1
        and _card_count(deck, "LPD_SHF") == 1
        and _card_count(deck, "LPD_BCH") == 0
    )


#: MASTER's fresh-loading card is ``f"F {batch:<2}  {rot}"``
#: (:meth:`lpopt.vendor.masterrl.domain.FuelItem.to_card`): the batch field is
#: TWO characters wide and ``:<2`` is a *minimum* width, so a longer id is
#: emitted verbatim and MASTER absorbs it without a diagnostic (defect 20260830).
_MAX_BATCH_ID_LEN = 2

#: ``F <batch> <rot>`` cells inside a ``%LPD_SHF`` body (comma-separated cards).
_SHF_FRESH_RE = re.compile(r"\bF\s+(\S+)\s+\d+", re.IGNORECASE)


def parse_lpd_bc_batch_ids(deck: str) -> tuple[str, ...]:
    """Batch ids declared in the deck's ``%LPD_B&C`` roster, in file order.

    ``&`` is not a ``%card`` name character, so the block is located by a prefix
    scan (as :func:`_parse_lpd_cx_fuel_order` does) rather than by
    :data:`_DECK_CARD_RE`.  Returns ``()`` when the deck declares no roster.
    """

    lines = deck.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().upper().startswith("%LPD_B&C")),
        None,
    )
    if start is None:
        return ()
    ids: list[str] = []
    for ln in lines[start + 1:]:
        if ln.lstrip().startswith("%"):
            break
        toks = ln.split("#", 1)[0].split()
        if toks and toks[0] != "/":
            ids.append(toks[0])
    return tuple(ids)


def validate_reload_deck(
    deck: str,
    restart_basename: str,
    *,
    expected_dims: tuple[int, int] = LIBRARY_DIMS,
    allowed_batch_ids: Iterable[str] | None = None,
) -> None:
    """Raise :class:`DeckValidationError` unless ``deck`` is a well-formed reload
    deck for ``restart_basename``.

    Checks (plan 5.2, the invariants a restart-read deck must satisfy so MASTER
    does not diverge to NaN):

    * ``%EXE_DEP`` present (>= 1 depletion step);
    * ``%EDT_OPT`` present (EOC restart write);
    * exactly one ``%LPD_SHF``;
    * ``%JOB_TYP`` present with ``irrst=1`` and *exactly one* ``MAS_RST.*``
      restart reference whose basename equals ``restart_basename``;
    * no ``%LPD_BCH`` (a fresh-core batch map has no place in a reload deck);
    * ``%GEN_DIM`` ``nbatch``/``ncomp`` equal to ``expected_dims`` (the library);
    * **every ``F <id>`` fresh card in ``%LPD_SHF`` names a batch id the deck
      itself declares** — a 2-char id present in ``%LPD_B&C`` (or in
      ``allowed_batch_ids`` when the roster is supplied out of band).

    The last check is the guard the HGD569 defect needed (memo 20260830 §6/R1):
    a verifier whose resolver carries an EMPTY ``type_id -> alias`` bridge leaves
    the pattern's full ``type_id`` labels in the SHF, MASTER finds no such batch
    in ``%LPD_B&C``, silently resolves every position to an unrelated batch, and
    a whole wave of decks is computed on a core nobody designed.  MASTER emits no
    error for this, so the deck must be refused HERE, before Popen.
    """

    problems: list[str] = []

    if _card_count(deck, "EXE_DEP") < 1:
        problems.append("no %EXE_DEP depletion steps (restart-read deck without a burnup chain)")
    if _card_count(deck, "EDT_OPT") < 1:
        problems.append("no %EDT_OPT (no EOC restart write)")
    shf = _card_count(deck, "LPD_SHF")
    if shf != 1:
        problems.append(f"expected exactly one %LPD_SHF, found {shf}")
    if _card_count(deck, "LPD_BCH"):
        problems.append("contains %LPD_BCH fresh-core batch map (not a reload deck)")

    typ_lines = _card_data_lines(deck, "JOB_TYP")
    if _card_count(deck, "JOB_TYP") != 1 or not typ_lines:
        problems.append("expected exactly one %JOB_TYP with a data line")
    else:
        first = typ_lines[0].split("#", 1)[0].split()
        irrst = first[0] if first else ""
        if irrst != "1":
            problems.append(f"%JOB_TYP irrst must be 1 (restart read), found {irrst!r}")
        refs = [ln.split("#", 1)[0].strip() for ln in typ_lines[1:]]
        rst_refs = [
            Path(ref).name for ref in refs if Path(ref).name.upper().startswith("MAS_RST.")
        ]
        if len(rst_refs) != 1:
            problems.append(
                f"expected exactly one MAS_RST.* restart reference in %JOB_TYP, "
                f"found {len(rst_refs)}"
            )
        elif rst_refs[0] != restart_basename:
            problems.append(
                f"%JOB_TYP restart reference {rst_refs[0]!r} != staged restart "
                f"{restart_basename!r}"
            )

    dim_lines = _card_data_lines(deck, "GEN_DIM")
    if not dim_lines:
        problems.append("missing %GEN_DIM")
    else:
        toks = dim_lines[0].split("#", 1)[0].split()
        if len(toks) < 5:
            problems.append("%GEN_DIM first data line has fewer than 5 fields")
        else:
            try:
                dims = (int(toks[3]), int(toks[4]))
            except ValueError:
                problems.append("%GEN_DIM nbatch/ncomp are not integers")
            else:
                if dims != tuple(expected_dims):
                    problems.append(
                        f"%GEN_DIM nbatch/ncomp {dims} != library "
                        f"{tuple(expected_dims)} (the reload template predates a "
                        f"library-dimension change; rebuild the band seeds against "
                        f"the CURRENT library, e.g. `lpopt design bootstrap --pair "
                        f"<A>_<B> --feed 121`, which rewrites bases/<folder> + "
                        f"cores/<folder>/bootstrap)"
                    )

    problems.extend(_shf_batch_problems(deck, allowed_batch_ids))

    if problems:
        raise DeckValidationError(
            "prepared reload deck failed sanity gate: " + "; ".join(problems)
        )


def _shf_batch_problems(
    deck: str, allowed_batch_ids: Iterable[str] | None
) -> list[str]:
    """``%LPD_SHF`` fresh-batch roster problems (see :func:`validate_reload_deck`).

    Two rules, deliberately independent:

    * an id longer than :data:`_MAX_BATCH_ID_LEN` is ALWAYS a defect — it cannot
      be a MASTER batch id, so it needs no roster to be recognised as one (this
      is the rule that fires on an untranslated ``type_id``);
    * an id absent from the roster is a defect whenever a roster is knowable
      (explicit ``allowed_batch_ids``, else the deck's own ``%LPD_B&C``).
    """

    if _card_count(deck, "LPD_SHF") != 1:
        return []                      # already reported by the caller
    try:
        body = extract_lpd_shf(deck)
    except Exception:                  # noqa: BLE001 — malformed span; caller reports
        return []
    fresh = [m.group(1) for m in _SHF_FRESH_RE.finditer(body)]
    if not fresh:
        return []

    roster = (
        tuple(str(b) for b in allowed_batch_ids)
        if allowed_batch_ids is not None
        else parse_lpd_bc_batch_ids(deck)
    )
    roster_set = {b.upper() for b in roster}

    oversized = sorted({b for b in fresh if len(b) > _MAX_BATCH_ID_LEN})
    unknown = sorted(
        {b for b in fresh if roster_set and b.upper() not in roster_set}
    ) if roster_set else []

    problems: list[str] = []
    if oversized:
        problems.append(
            f"%LPD_SHF fresh batch id(s) {oversized} exceed the {_MAX_BATCH_ID_LEN}-"
            f"character MASTER batch field: these are untranslated fuel type_ids, "
            f"not deck aliases — the emitting CaseAssetResolver has an EMPTY "
            f"registry alias bridge (pass registry_aliases= / build it from the "
            f"package's registry.json).  MASTER would absorb them silently"
        )
    if unknown:
        problems.append(
            f"%LPD_SHF fresh batch id(s) {unknown} are absent from the deck's "
            f"%LPD_B&C roster ({len(roster_set)} batch ids); MASTER resolves an "
            f"unknown batch id without a diagnostic, so the core would not be the "
            f"one this pattern describes"
        )
    return problems


# --------------------------------------------------------------------------- #
# cross-family reload-template synthesis (plan 5.2 / 12.1)
# --------------------------------------------------------------------------- #
#: Placeholder restart the synthesized template references.  ``prepare_cycle1_deck``
#: rewrites it to the real staged restart per case, so the cached template stays
#: restart-agnostic and is reused verbatim across restarts and campaigns.
_SYNTH_RESTART_PLACEHOLDER = "MAS_RST.SYNTH.00"


def synth_roster_for(library_id: str) -> tuple[str, ...]:
    """Full fuel-type roster used to synthesize a reload template for ``library_id``.

    ga80 -> the 80 letter types (A1..N6); each maps to an XS set ``FA_<type>`` the
    shipped ``MAS_XSL`` / ``MAS_HFF`` already carry, so a synthesized
    ``%LPD_B&C`` / ``%LPD_C&X`` / ``%LPD_HFF`` binds set names MASTER resolves
    against the same library the packaged decks use.  Unknown libraries return
    ``()`` (no built-in roster -> synthesis stays disabled unless an explicit
    ``synth_roster`` is supplied).
    """

    if library_id == "ga80":
        from ..data.fuel_types import GA80_TYPE_IDS

        return tuple(GA80_TYPE_IDS)
    return ()


def _parse_lpd_cx_fuel_order(deck: str) -> tuple[str, ...]:
    """Fuel aliases in ``%LPD_C&X`` composition-index order (index -> ``FA_<alias>``).

    Lets a synthesized deck reuse the vendor's exact composition-index -> set-name
    binding (the one the base restart was written under) instead of an arbitrary
    roster order.  ``&`` is not a ``%card`` name character, so the block is located
    by a prefix scan rather than :data:`_DECK_CARD_RE`.
    """

    lines = deck.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().upper().startswith("%LPD_C&X")),
        None,
    )
    if start is None:
        return ()
    order: list[tuple[int, str]] = []
    for ln in lines[start + 1:]:
        if ln.lstrip().startswith("%"):
            break
        toks = ln.split("#", 1)[0].split()
        if len(toks) >= 2 and toks[1].upper().startswith("FA_"):
            try:
                idx = int(toks[0])
            except ValueError:
                continue
            if idx > 0:
                order.append((idx, toks[1][3:]))
    order.sort()
    return tuple(name for _, name in order)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #
#: Package manifest carrying the ``{type_id: alias}`` bridge of a per-library
#: (paramA) design package.  ga80's FEASIBLE_PACKAGE has none (type_id == alias).
REGISTRY_FILENAME = "registry.json"


def registry_aliases_from_package(package_root: str | Path) -> dict[str, str]:
    """``{type_id: alias}`` from ``<package_root>/registry.json`` (``{}`` if absent).

    The single implementation of the alias bridge lookup:
    :func:`lpopt.search.resolver.paramA_registry_aliases` delegates here, and
    :class:`CaseAssetResolver` calls it for any caller that did not pass
    ``registry_aliases`` explicitly.  Unreadable / malformed manifests yield
    ``{}`` — a ga80 package legitimately has no registry, and a paramA package
    with a broken one is caught downstream by :func:`validate_reload_deck`, which
    refuses the emitted deck rather than trusting this map.
    """

    try:
        data = json.loads(
            (Path(package_root) / REGISTRY_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    aliases = data.get("aliases", {}) if isinstance(data, dict) else {}
    return (
        {str(t): str(a) for t, a in aliases.items()}
        if isinstance(aliases, dict)
        else {}
    )


class CaseAssetResolver:
    """Resolve restart + template deck for any ``(pair, feed)`` produce case."""

    def __init__(
        self,
        package_root: str | Path,
        promoted_root: str | Path | None = None,
        *,
        neutral_restart: str | Path | None = None,
        template_fallbacks: Sequence[str | Path] = (),
        fuel_library: Any = None,
        library_id: str = "ga80",
        synth_root: str | Path | None = None,
        synth_roster: Sequence[str] = (),
        synth_cycle: int = 12,
        registry_aliases: Mapping[str, str] | None = None,
        library_dims: tuple[int, int] = LIBRARY_DIMS,
        strict_restart: bool = False,
    ) -> None:
        self.package_root = Path(package_root)
        # When True, resolve() raises AssetResolutionError instead of returning a
        # level-3/4 cross-pair fallback restart (incompatible burnt types → MASTER
        # INITIALIZE failure).  Off by default so every existing caller — which
        # relies on the graceful e_core/neutral fallback — is byte-identical.
        self.strict_restart = bool(strict_restart)
        self.promoted_root = (
            Path(promoted_root) if promoted_root is not None else self.package_root / "promoted"
        )
        self.neutral_restart = Path(neutral_restart) if neutral_restart else None
        self.template_fallbacks = tuple(str(p) for p in template_fallbacks)
        self.fuel_library = fuel_library
        self.library_id = library_id
        # Synthesis fallback: enabled iff a cache root is set AND a roster is
        # available (explicit ``synth_roster`` or a built-in one for
        # ``library_id``).  Off by default so existing callers are unchanged.
        self.synth_root = Path(synth_root) if synth_root is not None else None
        self.synth_roster = tuple(str(t) for t in synth_roster) if synth_roster else ()
        self.synth_cycle = int(synth_cycle)
        # Per-library alias bridge (paramA packages name bases/cores folders and
        # deck batch ids by 2-char alias, while the curriculum keys pairs/patterns
        # by full type_id).  ``registry_aliases`` is the package registry's
        # ``{type_id: alias}`` map; ga80 (type_id == alias) passes ``None`` and is
        # unaffected.  ``type_to_alias`` translates a pattern's fresh batch labels
        # to the deck's %LPD_B&C ids at deck-emission; ``alias_to_type`` lets the
        # level-3 e_core scorer resolve an alias-named base folder against the
        # (type_id-keyed) fuel library.
        #
        # ``None`` (the DEFAULT, i.e. a caller that did not think about aliases)
        # now DERIVES the bridge from the package it was pointed at, because the
        # bridge is a property of that package and nothing else: it is exactly
        # ``package_root/registry.json``'s ``aliases`` map, which is what every
        # correct caller (``resolver.build_case_resolver``) was passing by hand.
        # Leaving it empty is what silently emitted 13-character type_ids into
        # %LPD_SHF for a whole wave (memo 20260830 §3; the verifier's own
        # fallback resolver was the caller that forgot).  An explicit ``{}``
        # still means "no bridge" for a caller that means it.
        if registry_aliases is None:
            registry_aliases = registry_aliases_from_package(self.package_root)
        self.type_to_alias: dict[str, str] = dict(registry_aliases) if registry_aliases else {}
        self.alias_to_type: dict[str, str] = {a: t for t, a in self.type_to_alias.items()}
        #: MASTER library (nbatch, ncomp) for this cell's library — read by the
        #: verifier so the reload-deck sanity gate checks the RIGHT %GEN_DIM.
        self.library_dims = tuple(library_dims)

    # -- restart catalog --------------------------------------------------- #
    def _catalog(self, package_root: Path, promoted_root: Path) -> list[_RestartEntry]:
        """Every available base restart keyed by ``(pair, feed)`` and source."""

        entries: list[_RestartEntry] = []
        for source, root in (("native", package_root / "bases"), ("promoted", promoted_root)):
            if not root.is_dir():
                continue
            for folder_path in sorted(root.iterdir()):
                if not folder_path.is_dir():
                    continue
                folder = folder_path.name
                pair = _pair_of_folder(folder)
                feed = _feed_of_folder(folder, pair)
                if feed is None:
                    continue
                restart = _only_restart(folder_path)
                if restart is not None:
                    entries.append(_RestartEntry(pair, feed, restart, source))
        return entries

    def _pair_e_core(self, pair: str) -> float | None:
        """Nominal equal-split core-average enrichment of a case (or ``None``).

        Handles a 2-type pair *and* a graded case of up to
        :data:`~lpopt.search.genome.MAX_FRESH_TYPES` members (``A_B_C``,
        ``A_B_C_D_E``, ...): the level-3 ladder rung ranks candidate restarts by
        ``|Δe_core|``, and a graded case's e_core is the composition mean over
        *all* its members (:func:`~lpopt.data.fuel_types.case_e_core`), so a
        graded cell resolves onto the nearest-reactivity *pair* restart instead
        of failing to score.  The 2-member path is the unchanged
        ``pair_e_core(a, b, 0.5)`` call.
        """

        if self.fuel_library is None:
            return None
        parts = [p for p in pair.split("_") if p]
        if not (2 <= len(parts) <= MAX_FRESH_TYPES):
            return None
        # Bridge alias-named base folders (paramA) to the type_id-keyed library.
        members = [self.alias_to_type.get(p, p) for p in parts]
        try:
            return float(_case_e_core(self.fuel_library, members, self.library_id))
        except (KeyError, ValueError, ZeroDivisionError):
            return None

    def _resolve_restart(
        self,
        case: CaseKey,
        catalog: list[_RestartEntry],
        neutral: Path | None,
    ) -> tuple[Path | None, int, list[str]]:
        notes: list[str] = []

        # Level 0 — native exact.
        for entry in catalog:
            if entry.source == "native" and entry.pair == case.pair and entry.feed == case.feed:
                return entry.path, 0, notes

        # Level 1 — promoted exact.
        for entry in catalog:
            if entry.source == "promoted" and entry.pair == case.pair and entry.feed == case.feed:
                notes.append("restart: promoted cache hit")
                return entry.path, 1, notes

        # Level 2 — same pair, nearest feed.
        same_pair = [e for e in catalog if e.pair == case.pair]
        if same_pair:
            best = min(same_pair, key=lambda e: (abs(e.feed - case.feed), e.feed))
            notes.append(f"restart: same pair, nearest feed {best.feed} (want {case.feed})")
            return best.path, 2, notes

        # Level 3 — nearest e_core pair, same-or-nearest feed.
        target_e = self._pair_e_core(case.pair)
        if target_e is not None:
            scored: list[tuple[float, _RestartEntry]] = []
            for entry in catalog:
                other_e = self._pair_e_core(entry.pair)
                if other_e is None:
                    continue
                scored.append((abs(other_e - target_e), entry))
            if scored:
                best_delta = min(delta for delta, _ in scored)
                near = [e for delta, e in scored if abs(delta - best_delta) < 1.0e-9]
                best = min(near, key=lambda e: (abs(e.feed - case.feed), e.feed))
                notes.append(
                    f"restart: nearest e_core pair {best.pair} "
                    f"(|Δe|={best_delta:.3f}), feed {best.feed} (want {case.feed})"
                )
                return best.path, 3, notes

        # Level 4 — configured neutral warm restart.
        if neutral is not None and neutral.exists():
            notes.append(f"restart: neutral warm restart {neutral.name}")
            return neutral, 4, notes

        notes.append("restart: NONE — no native/promoted/pair/e_core/neutral candidate")
        return None, -1, notes

    def _resolve_template(
        self, case: CaseKey, package_root: Path
    ) -> tuple[Path | None, list[str]]:
        notes: list[str] = []

        exact = sorted((package_root / "cores" / case.folder).glob("*/MAS_INP_cy*.inp"))
        same_pair = sorted(
            p
            for folder in (package_root / "cores").glob(f"{case.pair}*")
            if folder.is_dir() and _pair_of_folder(folder.name) == case.pair
            for p in folder.glob("*/MAS_INP_cy*.inp")
        )

        # Pass 1 — RELOAD decks only, in tier order (exact -> same-pair ->
        # configured fallbacks).  A reload deck is the *only* template MASTER can
        # restart-read without diverging; a cy1-style skeleton (with %LPD_BCH /
        # no %EXE_DEP) that happens to sort first must never win over a real
        # reload deck that also exists (plan 5.2).
        reload_hit = self._first_readable_reload(exact)
        if reload_hit is not None:
            notes.append("deck: exact case deck")
            return reload_hit, notes
        reload_hit = self._first_readable_reload(same_pair)
        if reload_hit is not None:
            notes.append("deck: same-pair fallback deck")
            return reload_hit, notes
        for pattern in self.template_fallbacks:
            reload_hit = self._first_readable_reload(sorted(_glob_anywhere(pattern)))
            if reload_hit is not None:
                notes.append(f"deck: template_fallback reload glob ({pattern})")
                return reload_hit, notes

        # Pass 2 — no reload deck available anywhere: fall back to any readable
        # deck (keeps synthetic/legacy single-cycle decks resolvable; the deck
        # sanity gate downstream still refuses a non-reload deck before Popen).
        readable = self._first_readable(exact)
        if readable is not None:
            notes.append("deck: exact case deck (non-reload)")
            return readable, notes
        if exact:
            notes.append("deck: exact case deck present but unreadable (dehydrated)")
        readable = self._first_readable(same_pair)
        if readable is not None:
            notes.append("deck: same-pair fallback deck (non-reload)")
            return readable, notes
        for pattern in self.template_fallbacks:
            readable = self._first_readable(sorted(_glob_anywhere(pattern)))
            if readable is not None:
                notes.append(f"deck: template_fallback glob ({pattern})")
                return readable, notes

        # Last resort: an existing-but-unreadable exact deck (best effort so the
        # resolver never returns None purely because of a dehydrated placeholder).
        if exact:
            notes.append("deck: best-effort unreadable exact deck")
            return exact[0], notes

        # Synthesis tier: nothing packaged resolved anywhere.  Build a restart-read
        # reload template from the library's composition tables (coredeck) so a
        # cross-family pair — which ships no deck — still receives a byte-compatible
        # template.  Cached under synth_root/<pair>/ and reused across campaigns.
        synth_path, synth_note = self._synthesize_template(case, package_root)
        if synth_path is not None:
            notes.append(synth_note)
            return synth_path, notes

        notes.append("deck: NONE — no exact/same-pair/fallback template found")
        return None, notes

    @staticmethod
    def _first_readable(paths: Iterable[Path]) -> Path | None:
        for path in paths:
            if path.is_file() and _is_readable(path):
                return path
        return None

    @staticmethod
    def _first_readable_reload(paths: Iterable[Path]) -> Path | None:
        """First readable deck in ``paths`` that is a genuine reload deck."""

        for path in paths:
            if not (path.is_file() and _is_readable(path)):
                continue
            try:
                text = _read_deck_flex(path)
            except OSError:
                continue
            if _is_reload_deck(text):
                return path
        return None

    # -- synthesis --------------------------------------------------------- #
    def _vendor_fuel_order(self, package_root: Path) -> tuple[str, ...]:
        """Fuel roster in the vendor's ``%LPD_C&X`` order, from any shipped deck.

        Every ga80 deck (reload or cy1) carries the same 80-type ``%LPD_C&X``
        table, so the first readable one fixes the canonical composition-index ->
        set-name binding.  ``()`` when the package ships no readable deck.

        Root-cause fix for the paramA 6.x-band production blocker: a package can
        carry a STALE bootstrap deck (e.g. ``cores/P0_P1/bootstrap`` written when
        the library had fewer fuel types) alongside current ones.  Because
        ``sorted(...)`` is alphabetical, such a stale deck could win and poison
        the whole synthesis roster (wrong ``%GEN_DIM``).  So we PREFER the first
        readable deck whose roster is DIMENSION-CONSISTENT with the package
        library (``self.library_dims``); a mismatched (stale) deck is only used as
        a last resort, and even then the downstream sanity gate rejects it with a
        clear rebuild hint.  ga80 is unaffected (all decks are 80-type == its
        dims).
        """

        from ..design.coredeck import library_dims

        cores = package_root / "cores"
        if not cores.is_dir():
            return ()
        first: tuple[str, ...] = ()          # first readable, dims aside (fallback)
        for deck_path in sorted(cores.glob("*/*/MAS_INP_cy*.inp")):
            if not (deck_path.is_file() and _is_readable(deck_path)):
                continue
            try:
                text = _read_deck_flex(deck_path)
            except OSError:
                continue
            order = _parse_lpd_cx_fuel_order(text)
            if not order:
                continue
            if not self.library_dims or library_dims(len(order)) == tuple(self.library_dims):
                return order                 # dimension-consistent -> trust it
            if not first:
                first = order                # remember a stale one as last resort
        return first

    def _effective_synth_roster(self, package_root: Path) -> tuple[str, ...]:
        """Roster to synthesize with: explicit override, else the vendor's own
        order (when it matches the library roster), else the built-in constant."""

        if self.synth_roster:
            return self.synth_roster
        known = synth_roster_for(self.library_id)
        vendor = self._vendor_fuel_order(package_root)
        if vendor and (not known or set(vendor) == set(known)):
            return vendor
        return known

    def _synthesize_template(
        self, case: CaseKey, package_root: Path
    ) -> tuple[Path | None, str | None]:
        """Synthesize (or reuse) a cached reload template for ``case``.

        Returns ``(path, note)`` or ``(None, None)`` when synthesis is disabled
        (no ``synth_root``) or unavailable (empty roster).  The cached deck is
        validated by :func:`validate_reload_deck` before it is trusted/written,
        so a synthesized or dehydrated-cache deck can never reach MASTER malformed.
        """

        if self.synth_root is None:
            return None, None
        roster = self._effective_synth_roster(package_root)
        if not roster:
            return None, None

        from ..design.coredeck import build_reload_deck, library_dims

        dims = library_dims(len(roster))
        cache_dir = self.synth_root / case.pair
        deck_path = cache_dir / f"MAS_INP_cy{self.synth_cycle:02d}.inp"

        # Cache reuse across campaigns: trust a prior synthesized deck only if it
        # still parses as a reload deck for this library (else rebuild it).
        if deck_path.is_file() and _is_readable(deck_path):
            try:
                cached = _read_deck_flex(deck_path)
                validate_reload_deck(
                    cached, _SYNTH_RESTART_PLACEHOLDER, expected_dims=dims
                )
            except (OSError, DeckValidationError):
                pass
            else:
                return deck_path, f"deck: synthesized (cache reuse, {case.pair})"

        deck = build_reload_deck(list(roster), _SYNTH_RESTART_PLACEHOLDER, self.synth_cycle)
        # Fail fast: never cache a deck the pre-Popen sanity gate would reject.
        validate_reload_deck(deck, _SYNTH_RESTART_PLACEHOLDER, expected_dims=dims)
        _atomic_write_text(deck_path, deck)
        return deck_path, (
            f"deck: synthesized ({case.pair}, {self.library_id}, {len(roster)} types)"
        )

    # -- public API -------------------------------------------------------- #
    def resolve(
        self,
        case_key: CaseKey | tuple[str, int],
        package_root: str | Path | None = None,
        promoted_root: str | Path | None = None,
    ) -> ResolvedAssets:
        """Resolve the best restart + template deck for ``case_key``."""

        case = _as_case_key(case_key)
        pkg = Path(package_root) if package_root is not None else self.package_root
        promoted = Path(promoted_root) if promoted_root is not None else self.promoted_root

        catalog = self._catalog(pkg, promoted)
        restart, level, restart_notes = self._resolve_restart(case, catalog, self.neutral_restart)
        template, template_notes = self._resolve_template(case, pkg)

        basename = restart.name if restart is not None else "none"
        kind = _LEVEL_KIND.get(level, "unresolved")
        provenance = f"{kind}:{basename}"
        # Strict guard (opt-in): a cross-pair fallback (level >= 3) restart is
        # burnt-type-incompatible with the target ga80 deck's %LPD_B&C.  Fail HARD
        # here — before Popen — instead of letting MASTER die at INITIALIZE and
        # charging the wasted call to the budget (forensic 20260721).
        if self.strict_restart and restart is not None and level >= 3:
            raise AssetResolutionError(
                f"{case.pair}/feed-{case.feed}: no pair-matched restart "
                f"(best is {kind} '{basename}', level {level}); its burnt types are "
                f"absent from the {case.pair} %LPD_B&C and MASTER would fail at "
                f"INITIALIZE.  Provide a native/promoted restart for this pair or "
                f"restrict the search to restart-bearing pairs.  Notes: "
                f"{'; '.join(restart_notes)}"
            )
        return ResolvedAssets(
            case_key=case,
            restart_path=restart,
            template_deck_path=template,
            fallback_level=level,
            restart_provenance=provenance,
            notes=tuple(restart_notes + template_notes),
        )

    def prepare_cycle1_deck(
        self, template_text: str, pattern: Pattern, restart_basename: str
    ) -> str:
        """Return a cycle-1 deck for ``pattern`` that references ``restart_basename``.

        Two edits, both byte-scoped:

        1. ``replace_lpd_shf`` swaps only the ``%LPD_SHF`` body for the pattern
           (vendor primitive; preserves every other character);
        2. ``advance_cycle_deck`` rewrites *only* the sole ``%JOB_TYP`` restart
           reference to ``restart_basename`` while keeping the deck's own
           ``%JOB_IDE`` cycle (so ``%JOB_IDE`` / a matching ``%JOB_TIT`` ``CYnn``
           title stay byte-identical).

        This mirrors how the vendor equilibrium chain advances a cycle
        (``equilibrium.EquilibriumRunner.run`` L513-518) — the restart reference
        is the single line MASTER validates against the staged ``MAS_RST.*``.

        For a per-library (paramA) package, the pattern carries full ``type_id``
        fresh-batch labels (the curriculum/model key), but the reload deck's
        ``%LPD_B&C`` batch ids are the 2-char MASTER aliases the coredeck built.
        The SHF loading is therefore translated ``type_id -> alias`` here so it
        matches the deck's batch ids; ga80 (``type_id == alias``, empty map) is a
        no-op and stays byte-identical.
        """

        pattern = self.alias_pattern(pattern)
        deck = replace_lpd_shf(template_text, pattern.to_shf())
        cycle = deck_cycle(deck)  # keep the deck's own cycle -> title stays put
        return advance_cycle_deck(deck, restart_basename, cycle)

    def alias_pattern(self, pattern: Pattern) -> Pattern:
        """Translate a pattern's fresh ``type_id`` batch labels to deck aliases.

        A no-op (returns ``pattern`` unchanged) when no ``registry_aliases`` were
        supplied (ga80) or no fresh label is a known ``type_id``.  Used both to
        stage the reload template and — because the vendor ``EquilibriumRunner``
        re-injects ``pattern.to_shf()`` on *every* cycle — to hand the runner an
        alias-space pattern so the SHF batch ids keep matching the deck's
        ``%LPD_B&C`` across the whole chain.
        """

        if not self.type_to_alias:
            return pattern
        changed = False
        items = []
        for item in pattern.items:
            if item.is_fresh and item.batch in self.type_to_alias:
                items.append(_dc_replace(item, batch=self.type_to_alias[item.batch]))
                changed = True
            else:
                items.append(item)
        return Pattern(tuple(items)) if changed else pattern

    def alias_case_key(self, case_key: CaseKey | tuple[str, int]) -> CaseKey:
        """Translate a case's ``type_id`` members to the deck's alias members.

        The vendor runner validates ``pattern.validate_case(key.pair, feed)``, so
        the case key handed to the runner must be in the SAME (alias) space as the
        :meth:`alias_pattern` it evaluates.  A no-op for ga80 (empty map) and for a
        case whose members are not known ``type_id``\\ s.  Handles a 2-type pair
        and a graded case of any width up to
        :data:`~lpopt.search.genome.MAX_FRESH_TYPES` identically (member-wise
        map, then re-join); ``validate_case`` reads the members as a SET, so the
        join order only has to match the deck's own ``case_pair`` string.
        """

        case = _as_case_key(case_key)
        if not self.type_to_alias:
            return case
        parts = case.pair.split("_")
        if not (2 <= len(parts) <= MAX_FRESH_TYPES) or not all(parts):
            return case
        mapped = [self.type_to_alias.get(p, p) for p in parts]
        if mapped == parts:
            return case
        return CaseKey("_".join(mapped), case.feed)

    def promote(self, case_key: CaseKey | tuple[str, int], final_restart_path: str | Path) -> Path:
        """Copy a converged chain's final ``MAS_RST.*`` into the promoted cache.

        The copy is atomic (temp file in the destination dir + ``os.replace``) and
        the destination folder is normalized to a single ``MAS_RST.*`` so a later
        level-1 resolution finds exactly one restart.
        """

        case = _as_case_key(case_key)
        source = Path(final_restart_path)
        if not source.is_file():
            raise FileNotFoundError(f"restart to promote does not exist: {source}")
        if not source.name.upper().startswith("MAS_RST."):
            raise ValueError(f"promoted restart must be a MAS_RST.* file: {source.name}")

        dest_dir = self.promoted_root / case.folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name

        data = source.read_bytes()
        fd, tmp_name = tempfile.mkstemp(prefix=f".{source.name}.", suffix=".tmp", dir=dest_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, dest)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

        # Keep the promoted folder to a single restart so ``_only_restart`` is
        # unambiguous on the next resolve.
        for other in dest_dir.glob("MAS_RST.*"):
            if other.name != dest.name:
                try:
                    other.unlink()
                except OSError:
                    pass
        return dest


def _glob_anywhere(pattern: str) -> list[Path]:
    """Filesystem glob supporting an absolute or drive-anchored ``pattern``."""

    p = Path(pattern)
    anchor = p.anchor
    if anchor:
        rel = pattern[len(anchor):]
        return list(Path(anchor).glob(rel))
    return list(Path().glob(pattern))


__all__ = [
    "AssetResolutionError",
    "CaseAssetResolver",
    "DeckValidationError",
    "LIBRARY_DIMS",
    "REGISTRY_FILENAME",
    "ResolvedAssets",
    "parse_lpd_bc_batch_ids",
    "registry_aliases_from_package",
    "synth_roster_for",
    "validate_reload_deck",
]
