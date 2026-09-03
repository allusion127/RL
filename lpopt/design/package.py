"""Assemble a paramA MASTER package in FEASIBLE_PACKAGE layout (plan 12.1).

Target layout (what ``[verify] package_root`` and ``MasterRunner`` require):

    <pkg>/
      designs.json                     design records (fuel_types ingest source)
      registry.json                    type_id <-> alias map
      hgc/FA_<alias>.HGC + .out        DeCART products (fuel_types MASS source)
      lib/MAS_XSL, lib/MAS_HFF         TotalBatcher library
      bases/<folder>/MAS_RST.*         band-seed restarts (written by bootstrap)
      cores/<folder>/<id>/MAS_INP_cyNN.inp   reload template (replace_lpd_shf-ready)

``ingest_fuel_types`` registers the new types into ``fuel_types.parquet`` with
``library_id="paramA"`` and full physics (``feature_poor=False``).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..data.fuel_types import FuelPaths, build_fuel_table, fuel_paths_from_config
from .coredeck import DEFAULT_CORE, CoreParams, build_reload_deck, library_dims
from .library import (
    LibraryBuild,
    build_master_library,
    default_tool_paths,
    require_snapshot,
)
from .spec import (
    GD_CARRIER_ENR,
    OCTANT_ROWS,
    DesignRegistry,
    FuelDesign,
    format_gd_positions,
)
from .spec import parse_gd_positions as _spec_parse_gd_positions


@dataclass
class DesignSource:
    """A produced lattice: its design, alias, and harvested product files.

    ``sum_path`` and ``deck_path`` are the two audit channels the package used to
    lose (task #13b): ``data/design/package/hgc/`` held 37 ``.HGC`` + 37 ``.out``
    and nothing else, so ``fuel_types``' cond_v4 harvest had no ``.sum`` to read and
    ``zone_pin_count`` stayed NaN for want of a staged ``dec_FA_*.inp``.
    ``deck_path`` must be the AUTHORED deck (``DecartRun.deck_path`` or the entry in
    the template tree) — never the working directory's ``decart.inp``, which
    :func:`lpopt.design.lattice.harvest` deletes.
    """

    design: FuelDesign
    alias: str
    hgc_path: Path
    out_path: Path | None = None
    sum_path: Path | None = None
    deck_path: Path | None = None


# --------------------------------------------------------------------------- #
# designs.json schema (task #13)
# --------------------------------------------------------------------------- #
#: The five design axes + the identity/registry fields every record has carried
#: since the first paramA package.  Written for EVERY record.
DESIGN_BASE_FIELDS: tuple[str, ...] = (
    "type_id", "e1", "e2", "zoning_variant", "gd_wt", "n_gd", "alias", "gd_u_enr",
)

#: Optional record fields.  A record that carries none of them is byte-identical
#: to what the pre-#13 writer emitted, so every existing package/caller is
#: unchanged; a record that carries any is an AUTHORED on-demand type.
#:
#: ``gd_positions``   octant Gd pin map, ``"r:c;r:c;..."`` (0-indexed) — the
#:                    audit anchor promoted from optional to mandatory-for-
#:                    authored rows (design v2 sec. 6.5).
#: ``layout``         the authored layout string the pin map was rendered from.
#: ``base_template``  path of the frozen ``dec_FA_*.inp`` it was authored on top
#:                    of (provenance for the byte diff).
#: ``xenon_mode``     DeCART xenon treatment (``"TR"`` for the equilibrium chain).
#: ``density``        fuel pellet density [g/cc] the lattice was run at.
#: ``provenance``     free text: which run/slice authored the row.
#: ``screen_*``       surrogate screening record (``screen_pattern`` is ``PA``/
#:                    ``PB`` — z1 screens as **PB**, see task #4).
#: ``decart_wall_s``  DeCART wall time [s] of the producing lattice run.
#: ``hgc_sha256`` / ``deck_sha256``  staged product hashes.
#: ``lat1600_*``     incumbent provenance the SHIPPED manifest already carries on
#:                   its four authored rows (``lat1600_id`` ``"Y1"``,
#:                   ``lat1600_role`` the reactivity-matching note).  Whitelisted
#:                   so the live manifest round-trips instead of being refused as
#:                   an unknown field / silently dropped on the next assembly.
DESIGN_OPTIONAL_FIELDS: tuple[str, ...] = (
    "gd_positions", "layout", "base_template", "xenon_mode", "density",
    "provenance", "screen_ff", "screen_k0", "screen_crossing_bu",
    "screen_model_sha", "screen_pattern", "decart_wall_s",
    "hgc_sha256", "deck_sha256", "lat1600_id", "lat1600_role",
)

#: Optional fields that are PURE PROVENANCE: present on incumbent rows, they say
#: nothing about whether the row was authored on demand.
_PROVENANCE_ONLY_FIELDS: tuple[str, ...] = ("lat1600_id", "lat1600_role")

#: Presence of ANY of these marks a record as an authored on-demand type, for
#: which ``gd_positions`` is mandatory: the pin map is the only thing that makes
#: two rows with the same quantized ``type_id`` distinguishable (design v2 sec.
#: 1.1 — ``type_id`` folds 4.6750 and 4.70 onto the same ``47``).  It is DERIVED
#: from the optional schema rather than hand-listed, so a screening-only row
#: (``screen_ff``/``screen_k0``/``hgc_sha256``, exactly the extras prereg line
#: 354 writes) cannot slip through without a pin map.
_AUTHORED_MARKERS: tuple[str, ...] = tuple(
    f for f in DESIGN_OPTIONAL_FIELDS
    if f != "gd_positions" and f not in _PROVENANCE_ONLY_FIELDS
)

class DesignManifestError(ValueError):
    """``designs.json`` record violates the schema (task #13)."""


def normalize_gd_positions(value) -> str:
    """Canonical ``"r:c;r:c"`` octant Gd pin map.

    Cells are 0-indexed octant coordinates, ``0 <= col <= row < 8`` — the
    convention of :func:`lpopt.design.spec.parse_gd_positions` and of the deck's
    own assembly triangle (row 0 is the assembly centre).  Accepts the canonical
    string itself or any iterable of ``(row, col)`` pairs (what
    ``FuelDesign.gd_positions`` carries).  The result is SORTED, so one layout has
    exactly one spelling here and in ``FuelDesign.gd_layout``/``layout_tag``.

    There is exactly ONE parser: :func:`lpopt.design.spec.parse_gd_positions`,
    the same one ``FuelDesign.__post_init__`` and ``layout_tag`` go through.  This
    wrapper only re-raises its ``ValueError`` as :class:`DesignManifestError` and
    adds the octant-triangle bound, so a manifest spelling can never disagree with
    the design's own.
    """
    try:
        pairs = _spec_parse_gd_positions(value)
    except ValueError as exc:
        raise DesignManifestError(
            f"gd_positions must be 'r:c;r:c;...' (0-indexed) or an iterable of "
            f"(row, col) pairs, got {value!r}: {exc}") from exc
    if not pairs:
        raise DesignManifestError("gd_positions must not be empty")
    for r, c in pairs:
        if not (0 <= c <= r < OCTANT_ROWS):
            raise DesignManifestError(
                f"gd_positions cell {r}:{c} is outside the {OCTANT_ROWS}-row "
                f"octant triangle (require 0 <= col <= row < {OCTANT_ROWS})")
    if len(set(pairs)) != len(pairs):
        raise DesignManifestError(f"gd_positions has duplicate cells: {pairs}")
    return format_gd_positions(pairs)


def parse_gd_positions(value: str | None) -> tuple[tuple[int, int], ...]:
    """Inverse of :func:`normalize_gd_positions` (``()`` for ``None``).

    Same result as :func:`lpopt.design.spec.parse_gd_positions` (sorted, 0-indexed)
    plus this module's validation.
    """
    if value is None:
        return ()
    return _spec_parse_gd_positions(normalize_gd_positions(value))


def design_record(source: "DesignSource", *, extra: Mapping | None = None,
                  require_gd_positions: bool = False) -> dict:
    """One ``designs.json`` record for ``source`` (schema of task #13).

    Optional fields come from ``extra`` first, then from attributes of the
    :class:`~lpopt.design.spec.FuelDesign` itself (so an authored design that
    carries ``gd_positions`` needs no out-of-band map), and a field that resolves
    to ``None`` is OMITTED — an unauthored record is byte-identical to the
    pre-#13 one.  Raises :class:`DesignManifestError` if the record is authored
    (or ``require_gd_positions``) but carries no ``gd_positions``.
    """
    rec = source.design.as_dict()
    # Optional fields are re-emitted below in schema order (and only when set),
    # so an ``as_dict`` that already carries one (``gd_positions`` once task #1
    # promotes it onto FuelDesign) neither duplicates nor emits a null.
    design_optional = {k: rec.pop(k) for k in DESIGN_OPTIONAL_FIELDS if k in rec}
    rec["alias"] = source.alias
    extra = dict(extra or {})
    gd_u = extra.pop("gd_u_enr", None)
    if gd_u is None:
        gd_u = getattr(source.design, "gd_u_enr", None)
    rec["gd_u_enr"] = float(GD_CARRIER_ENR if gd_u is None else gd_u)

    unknown = sorted(set(extra) - set(DESIGN_OPTIONAL_FIELDS) - set(DESIGN_BASE_FIELDS))
    if unknown:
        raise DesignManifestError(
            f"unknown designs.json field(s) for {rec['type_id']}: {unknown}")

    for name in DESIGN_OPTIONAL_FIELDS:
        value = extra.get(
            name, design_optional.get(name, getattr(source.design, name, None)))
        if value is None:
            continue
        rec[name] = normalize_gd_positions(value) if name == "gd_positions" else value

    # ``screen_pattern`` is DEFINED by the zoning variant (z1 -> PB, z2 -> PA);
    # writing the other one is silently wrong (the surrogate just evaluates a
    # different assembly), so the record is checked against the design itself.
    pattern = rec.get("screen_pattern")
    if pattern is not None:
        from .screen import pattern_for

        want = pattern_for(source.design.zoning_variant)
        if str(pattern) != want:
            raise DesignManifestError(
                f"design {rec['type_id']} is {source.design.zoning_variant} and "
                f"therefore screens as {want}, but the record says "
                f"screen_pattern={pattern!r}")

    authored = any(rec.get(m) is not None for m in _AUTHORED_MARKERS)
    if (authored or require_gd_positions) and not rec.get("gd_positions"):
        raise DesignManifestError(
            f"design {rec['type_id']} is an authored on-demand type "
            f"(fields: {[m for m in _AUTHORED_MARKERS if rec.get(m) is not None]}) "
            f"but carries no gd_positions; the pin map is a REQUIRED field "
            f"(design v2 sec. 6.5)")
    return rec


def load_designs_manifest(path: str | Path) -> dict:
    """Read ``designs.json`` with every optional field defaulted to ``None``.

    Backward compatible by construction: a manifest written before task #13
    (37 rows, ``gd_positions`` on 4 of them) loads with the missing fields as
    ``None`` rather than raising.  Returns
    ``{"library_id": ..., "designs": [record, ...]}``.
    """
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8-sig"))
    raw = doc.get("designs", doc) if isinstance(doc, dict) else doc
    library_id = doc.get("library_id", "paramA") if isinstance(doc, dict) else "paramA"
    records = []
    for r in raw:
        rec = dict(r)
        for name in DESIGN_OPTIONAL_FIELDS:
            rec.setdefault(name, None)
        records.append(rec)
    return {"library_id": library_id, "designs": records}


def write_designs_manifest(pkg_dir: str | Path, sources: list[DesignSource],
                           registry: DesignRegistry, *,
                           extras: Mapping[str, Mapping] | None = None,
                           require_gd_positions: bool = False) -> Path:
    """Write ``designs.json`` (fuel_types ingest source) + ``registry.json``.

    ``extras`` maps ``type_id`` -> the optional record fields of task #13
    (:data:`DESIGN_OPTIONAL_FIELDS`).  With no ``extras`` and plain designs the
    emitted JSON is byte-identical to the pre-#13 writer's.
    """
    pkg = Path(pkg_dir)
    pkg.mkdir(parents=True, exist_ok=True)
    extras = dict(extras or {})
    records = [
        design_record(s, extra=extras.get(s.design.type_id),
                      require_gd_positions=require_gd_positions)
        for s in sources
    ]
    manifest = pkg / "designs.json"
    manifest.write_text(
        json.dumps({"library_id": "paramA", "designs": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    registry.save(pkg / "registry.json")
    return manifest


def _copy_companion(src: Path | None, dst: Path) -> None:
    """Copy a harvested companion file into the package (no-op when absent)."""
    if src is None:
        return
    src = Path(src)
    if not src.is_file() or src.resolve() == dst.resolve():
        return
    shutil.copyfile(src, dst)


def stage_hgc(pkg_dir: str | Path, sources: list[DesignSource]) -> list[Path]:
    """Copy each ``FA_<alias>.HGC`` and its companions into ``<pkg>/hgc/``.

    Companions (task #13b), each staged only when the source carries it:

    * ``FA_<alias>.out``  — the MASS(g) inventory the fuel_types ingest parses;
    * ``FA_<alias>.sum``  — the reference k-inf curve + BOC coefficients (the 181
      queue already leaves one behind; ``harvest`` now keeps it too);
    * ``dec_FA_<alias>.inp`` — the AUTHORED deck, so the realized pin map is
      auditable as BYTES rather than as the ``gd_positions`` prose in
      ``designs.json`` (and ``zone_pin_count`` stops being NaN).

    Returns the staged ``.HGC`` paths (unchanged contract for the library build).
    """
    hgc_dir = Path(pkg_dir) / "hgc"
    hgc_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for s in sources:
        dst = hgc_dir / f"FA_{s.alias}.HGC"
        if Path(s.hgc_path).resolve() != dst.resolve():
            shutil.copyfile(s.hgc_path, dst)
        staged.append(dst)
        _copy_companion(s.out_path, hgc_dir / f"FA_{s.alias}.out")
        _copy_companion(s.sum_path, hgc_dir / f"FA_{s.alias}.sum")
        _copy_companion(s.deck_path, hgc_dir / f"dec_FA_{s.alias}.inp")
    return staged


def build_library_from_sources(pkg_dir: str | Path, sources: list[DesignSource],
                               apr1400_root: str | Path, *,
                               tools: dict | None = None,
                               snapshot_dir: str | Path | None = None) -> LibraryBuild:
    """Stage HGCs and run TotalBatcher into ``<pkg>/lib/``.

    ``snapshot_dir`` gates the rebuild on a pre-rebuild snapshot (task #12): the
    build proceeds only while ``<snapshot_dir>`` still hashes this package.
    Omitted (the default) the behaviour is byte-identical to before.
    """
    pkg = Path(pkg_dir)
    if snapshot_dir is not None:
        require_snapshot(pkg, snapshot_dir)
    hgc_paths = stage_hgc(pkg, sources)
    tp = tools or default_tool_paths(apr1400_root)
    return build_master_library(
        hgc_paths, pkg / "lib",
        mas_ref=tp["mas_ref"], prolog_exe=tp["prolog_exe"],
        totalbatcher_exe=tp["totalbatcher_exe"], library_id="paramA",
    )


def write_core_template(pkg_dir: str | Path, pair: str, feed: int,
                        aliases: list[str], restart_basename: str, *,
                        seed_id: str = "seed", cycle: int = 12,
                        core: CoreParams = DEFAULT_CORE) -> Path:
    """Write a reload template ``cores/<folder>/<seed_id>/MAS_INP_cyNN.inp``.

    ``restart_basename`` is the base restart the deck references; the harness
    ``prepare_cycle1_deck`` rewrites it to the resolved restart at eval time, and
    ``replace_lpd_shf`` overwrites the placeholder loading.
    """
    from ..vendor.masterrl.domain import CaseKey

    folder = CaseKey(pair, int(feed)).folder
    seed_dir = Path(pkg_dir) / "cores" / folder / seed_id
    seed_dir.mkdir(parents=True, exist_ok=True)
    deck = build_reload_deck(aliases, restart_basename, cycle, core=core)
    path = seed_dir / f"MAS_INP_cy{cycle:02d}.inp"
    path.write_text(deck, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# package regeneration after a roster change (task #13c)
# --------------------------------------------------------------------------- #
_CY_RE = re.compile(r"MAS_INP_cy(\d+)\.inp$", re.IGNORECASE)
_RST_RE = re.compile(r"\bMAS_RST\.[^\s#]+")
#: ``%GEN_DIM`` data line: ``nx ny nz nbatch ncomp``.
_GEN_DIM_RE = re.compile(r"^\s*%GEN_DIM\s*$", re.IGNORECASE)


def _deck_gen_dim(deck: str) -> tuple[int, int] | None:
    """``(nbatch, ncomp)`` off a deck's ``%GEN_DIM`` first data line."""
    lines = deck.splitlines()
    for i, line in enumerate(lines):
        if not _GEN_DIM_RE.match(line):
            continue
        for data in lines[i + 1:]:
            body = data.split("#", 1)[0].split()
            if not body:
                continue
            if body[0].startswith("%"):
                return None
            if len(body) >= 5 and all(t.lstrip("-").isdigit() for t in body[:5]):
                return int(body[3]), int(body[4])
            return None
    return None


def _deck_restart_basename(deck: str) -> str | None:
    """The ``MAS_RST.*`` name a reload template reads (``%JOB_TYP``)."""
    match = _RST_RE.search(deck)
    return match.group(0) if match else None


@dataclass
class CoreTemplatePlan:
    """One ``cores/<folder>/<seed>/MAS_INP_cyNN.inp`` slated for regeneration."""

    path: Path
    pair: str
    feed: int
    seed_id: str
    cycle: int
    restart_basename: str
    old_dims: tuple[int, int] | None
    new_dims: tuple[int, int]
    written: bool = False

    @property
    def stale(self) -> bool:
        return self.old_dims != self.new_dims

    def as_dict(self) -> dict:
        return {"path": str(self.path), "pair": self.pair, "feed": self.feed,
                "seed_id": self.seed_id, "cycle": self.cycle,
                "restart_basename": self.restart_basename,
                "old_dims": list(self.old_dims) if self.old_dims else None,
                "new_dims": list(self.new_dims), "stale": self.stale,
                "written": self.written}


@dataclass
class RegenReport:
    """Result (or dry-run listing) of :func:`regenerate_core_templates`."""

    new_dims: tuple[int, int]
    n_aliases: int
    templates: list[CoreTemplatePlan] = field(default_factory=list)
    purged: list[Path] = field(default_factory=list)
    #: ``bases/<folder>/MAS_RST.*`` restarts produced against the OLD library
    #: (task #13c item 3).  Regenerating ``cores/`` makes every template pass
    #: ``validate_reload_deck`` again while it still reads these — the one hard
    #: stop that would have caught the stale package is exactly what the
    #: regeneration removes, so they are enumerated and refused by default.
    stale_bases: list[Path] = field(default_factory=list)
    dry_run: bool = False

    @property
    def folders(self) -> list[str]:
        return sorted({t.path.parent.parent.name for t in self.templates})

    @property
    def stale(self) -> list[CoreTemplatePlan]:
        return [t for t in self.templates if t.stale]

    @property
    def stale_base_folders(self) -> list[str]:
        """The ``bases/<folder>`` names that must be re-bootstrapped (#13c-3)."""
        return sorted({p.parent.name for p in self.stale_bases})

    def as_dict(self) -> dict:
        return {"new_dims": list(self.new_dims), "n_aliases": self.n_aliases,
                "dry_run": self.dry_run, "folders": self.folders,
                "n_stale": len(self.stale),
                "templates": [t.as_dict() for t in self.templates],
                "purged": [str(p) for p in self.purged],
                "stale_bases": [str(p) for p in self.stale_bases],
                "stale_base_folders": self.stale_base_folders}


def core_template_paths(pkg_dir: str | Path) -> list[Path]:
    """Every ``cores/<folder>/<seed_id>/MAS_INP_cyNN.inp`` in the package."""
    cores = Path(pkg_dir) / "cores"
    if not cores.is_dir():
        return []
    return sorted(p for p in cores.glob("*/*/MAS_INP_cy*.inp") if p.is_file())


def stale_base_restarts(pkg_dir: str | Path) -> list[Path]:
    """Every ``bases/<folder>/MAS_RST.*`` in the package (task #13c item 3).

    A restart is keyed to the library it was produced against, so a roster change
    invalidates all of them; regenerating ``cores/`` alone yields a package that
    validates and is still wrong.
    """
    bases = Path(pkg_dir) / "bases"
    if not bases.is_dir():
        return []
    return sorted(p for p in bases.glob("*/MAS_RST.*") if p.is_file())


class StaleBasesError(RuntimeError):
    """``bases/`` still holds restarts built against the OLD library (#13c-3)."""

    def __init__(self, message: str, report: "RegenReport") -> None:
        super().__init__(message)
        self.report = report


def regenerate_core_templates(pkg_dir: str | Path, aliases: Sequence[str], *,
                              dry_run: bool = False,
                              core: CoreParams = DEFAULT_CORE,
                              synth_root: str | Path | None = None,
                              purge_synth: bool = True,
                              accept_stale_bases: bool = False) -> RegenReport:
    """Rewrite every ``cores/`` reload template against a NEW library roster.

    Adding types to the library changes ``%GEN_DIM`` (``nbatch``/``ncomp`` are
    ``3+N`` / ``5+N``), which makes EVERY template on disk stale: the deck a
    campaign resolves from ``cores/`` is used without a dimension check
    (``assets._resolve_template``), while ``validate_reload_deck`` refuses a
    mismatched ``%GEN_DIM`` — i.e. an unregenerated package hard-fails every
    existing pair before Popen.  This is the step the v1/v2 drafts omitted.

    Each template keeps its folder, seed id, cycle number and restart basename;
    only the roster (``%LPD_B&C`` / ``%LPD_C&X`` / ``%LPD_HFF`` / ``%GEN_DIM``)
    is rebuilt, from ``aliases`` (pass ``bootstrap.library_aliases(pkg)`` — the
    assembled ``lib/MAS_XSL`` COMP order).  ``synth_root`` (typically
    ``data/design/synth_decks``) is purged of its cached synthesized decks,
    which are keyed by pair and carry the OLD dimensions.

    ``dry_run=True`` writes and deletes nothing and returns the same report, so
    the regeneration can be listed and reviewed before it runs.

    Item 3 of task #13c — re-bootstrapping ``bases/`` — is NOT automated here (it
    is 8 of 9 MASTER chains, hours of compute on 199), but it cannot be forgotten
    silently either: :attr:`RegenReport.stale_bases` enumerates the restarts that
    the roster change invalidates and a non-dry run REFUSES to proceed while any
    exist unless ``accept_stale_bases=True``.  The refusal happens before any
    file is written, so a package is never left half-regenerated.

    ``purge_synth`` (the mandatory item 2) now needs ``synth_root``: the old
    default silently purged nothing and reported success.
    """
    # ONE reader of the ``<pair>[_f<feed>]`` folder convention: the resolver's,
    # so a regenerated template can never land in a folder the resolver would
    # then read as a different (pair, feed).  Imported lazily to leave the
    # design->search import graph unchanged.
    from ..search.assets import _feed_of_folder, _pair_of_folder

    pkg = Path(pkg_dir)
    aliases = [str(a) for a in aliases]
    if not aliases:
        raise ValueError("regenerate_core_templates needs a non-empty roster")
    if purge_synth and synth_root is None:
        raise ValueError(
            "purge_synth=True needs synth_root (typically "
            "data/design/synth_decks); pass purge_synth=False to skip the "
            "cached-deck purge deliberately")
    new_dims = library_dims(len(aliases))
    report = RegenReport(new_dims=new_dims, n_aliases=len(aliases), dry_run=dry_run,
                         stale_bases=stale_base_restarts(pkg))
    if report.stale_bases and not dry_run and not accept_stale_bases:
        raise StaleBasesError(
            f"regenerating cores/ to {new_dims} would leave "
            f"{len(report.stale_bases)} bases/ restart(s) built against the OLD "
            f"library, in folder(s) {report.stale_base_folders} — task #13c item "
            f"3 requires re-bootstrapping them.  Nothing was written.  Pass "
            f"accept_stale_bases=True once the re-bootstrap is scheduled.",
            report)

    for path in core_template_paths(pkg):
        seed_dir = path.parent
        folder = seed_dir.parent.name
        pair = _pair_of_folder(folder)
        feed = _feed_of_folder(folder, pair)
        if feed is None:
            continue
        deck = path.read_text(encoding="utf-8", errors="replace")
        restart = _deck_restart_basename(deck)
        if restart is None:
            raise ValueError(f"core template {path} names no MAS_RST.* restart")
        cy_match = _CY_RE.search(path.name)
        cycle = int(cy_match.group(1)) if cy_match else 12
        plan = CoreTemplatePlan(
            path=path, pair=pair, feed=feed, seed_id=seed_dir.name, cycle=cycle,
            restart_basename=restart, old_dims=_deck_gen_dim(deck),
            new_dims=new_dims,
        )
        if not dry_run:
            written = write_core_template(pkg, pair, feed, aliases, restart,
                                          seed_id=seed_dir.name, cycle=cycle,
                                          core=core)
            if written.resolve() != path.resolve():   # pragma: no cover - defensive
                raise ValueError(
                    f"regeneration wrote {written}, expected {path}")
            plan.written = True
        report.templates.append(plan)

    if purge_synth and synth_root is not None:
        root = Path(synth_root)
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    report.purged.append(p)
            if not dry_run:
                shutil.rmtree(root, ignore_errors=True)
                root.mkdir(parents=True, exist_ok=True)
    return report


def ingest_fuel_types(pkg_dir: str | Path, *, cfg=None, base_paths: FuelPaths | None = None,
                      store_path: str | Path | None = None):
    """Merge paramA rows into ``fuel_types.parquet`` (full physics).

    Uses ``build_fuel_table`` with ``paramA_root`` set to ``pkg_dir`` (its
    ``designs.json`` + ``hgc/FA_*.out`` drive the ingest).  Either ``cfg`` (an
    LpoptConfig) or ``base_paths`` supplies the other five libraries' locations.
    Returns the rebuilt DataFrame.

    The paramA rows produced by ``paramA_rows`` now also auto-fill the cond_v4
    physics columns from the staged ``hgc/FA_<alias>.sum`` / ``.HGC`` (reference
    k-inf curve + BOC reactivity coefficients + 2-group xs / ADF / pin-power
    peaking); ``zone_pin_count`` stays NaN unless a ``dec_FA_*.inp`` is staged.
    """
    pkg = Path(pkg_dir)
    if base_paths is None:
        if cfg is None:
            raise ValueError("ingest_fuel_types needs cfg or base_paths")
        base_paths = fuel_paths_from_config(cfg)
    store = Path(store_path) if store_path is not None else base_paths.store
    paths = FuelPaths(
        apr1400_root=base_paths.apr1400_root,
        ga80_hgc=base_paths.ga80_hgc,
        manual_yaml=base_paths.manual_yaml,
        store=store,
        paramA_root=pkg,
    )
    return build_fuel_table(paths, persist=True)


def assemble_package(pkg_dir: str | Path, sources: list[DesignSource],
                     registry: DesignRegistry, apr1400_root: str | Path, *,
                     tools: dict | None = None,
                     snapshot_dir: str | Path | None = None,
                     require_gd_positions: bool = False) -> LibraryBuild:
    """One call: manifest + registry + hgc staging + TotalBatcher library.

    Bootstrap (bases/) and core templates (cores/) are added by the caller once
    seeds exist, since they need an assembled ``lib/`` first.

    ``snapshot_dir`` (task #12) is checked FIRST — before the manifest is
    rewritten — because ``designs.json``/``registry.json`` are themselves
    snapshot members, so a gate placed after the rewrite could never pass.
    """
    if snapshot_dir is not None:
        require_snapshot(pkg_dir, snapshot_dir)
    write_designs_manifest(pkg_dir, sources, registry,
                           require_gd_positions=require_gd_positions)
    return build_library_from_sources(pkg_dir, sources, apr1400_root, tools=tools)


__all__ = [
    "DESIGN_BASE_FIELDS",
    "DESIGN_OPTIONAL_FIELDS",
    "CoreTemplatePlan",
    "DesignManifestError",
    "DesignSource",
    "RegenReport",
    "StaleBasesError",
    "assemble_package",
    "build_library_from_sources",
    "core_template_paths",
    "design_record",
    "ingest_fuel_types",
    "load_designs_manifest",
    "normalize_gd_positions",
    "parse_gd_positions",
    "regenerate_core_templates",
    "stage_hgc",
    "stale_base_restarts",
    "write_core_template",
    "write_designs_manifest",
]
