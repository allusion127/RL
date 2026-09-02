"""Retro-label ``f_xy`` / ``f_xya`` from ``MAS_OUT`` files still on disk.

::

    python -m lpopt.tools.backfill_fxy scan  --root runs --out scan.csv
    python -m lpopt.tools.backfill_fxy apply --csv scan.csv [--dry-run]

``F_xy`` is absent from ``MAS_SUM``, so the only retro-label source is a
``MAS_OUT`` that survived a run (design 20260829 §2).  ``scan`` walks a runs tree
and emits one CSV row per MASTER work dir; ``apply`` joins that CSV to
``records.parquet`` and fills the two nullable columns.  The two halves are split
so ``scan`` can be shipped to a campaign box (HOST_199) and only its ~100 kB CSV
returned instead of ~620 MB of MAS_OUT: **``scan`` touches no store, imports no
pandas/pyarrow/numpy** — the heavy imports live inside :func:`apply` alone.

Which work dirs are FINAL equilibrium cycles (the adjudication rule)
-------------------------------------------------------------------
Only the final cycle of a converged chain is an equilibrium label.  Local runs
trees are full of other cycles that merely survived a failed Windows ``rmtree``
(design §2.1: 54% of a 400-dir sample were the chain's FIRST cycle), so every dir
is adjudicated from evidence inside itself and its siblings, and anything not
proven final is recorded with its reason and NOT used:

``nonfinite``    the ``NONFINITE_FLUX`` sentinel is present — a physics kill,
                 whose FXYP describes a diverging solve (§5.2).
``no_mas_sum``   no ``MAS_SUM``: a failed cycle dir, trimmed by
                 ``verify._trim_failed_work_dir`` to its MAS_OUT tail.
``no_digest``    the dir name is not ``<digest16>__<restart_tag>`` (the shape
                 ``verify.py:931`` stages), so it carries no join key.
``first_cycle``  ``MAS_INP``'s restart reference still names the SEED restart
                 from the dir name, i.e. this is cycle 1 of a chain whose later
                 cycles are gone.  Every later cycle reads a restart MASTER
                 itself generated, so the two names differ.
``superseded``   a sibling dir with the same ``<digest16>__<tag>`` prefix has a
                 higher ``%JOB_IDE`` cycle: this one is an earlier cycle of the
                 same chain that outlived its purge.
``final``        none of the above — harvested.

:func:`is_final_evidence` is the single test for "harvestable"; the archive
layouts below add their own evidence strings to it.

``%JOB_IDE`` is a chain-POSITION signal only *between siblings*: it starts at the
template deck's own cycle number, not at the seed restart's, so its absolute
value says nothing (measured: a verified final cycle with seed ``APRQ_10`` and
``%JOB_IDE 12``).  The one absolute cross-check available is EFPD: the largest
``$P2D`` step EFPD equals the dir's ``MAS_SUM`` cycle length on 89/89 retained
final cycles, so :func:`apply` refuses any row whose ``efpd_max`` disagrees with
the store row's ``cyclen`` by more than :data:`CYCLEN_TOL_EFPD` — a mid-chain dir
that slipped through the rules above has a different cycle length and is caught
there.

Archived trees (``2_LP/LOW_Fr_MASTER_result``)
----------------------------------------------
The raw-output archive is the same MASTER work dirs after a collection sweep, and
the sweep broke both halves of the rules above:

* **Flattened case names.**  Origins ``local_3GA`` / ``local_5RL`` / ``srv181`` /
  ``srv198`` / ``srv199`` store each case as ONE directory whose name is the
  original nested path compressed with a ``~`` elision, so the ``<digest16>__``
  prefix is gone (0 of 11,483 collected case names keep it) and the sibling
  worker directory that ``superseded`` compares within is gone too.  Both are
  recovered, not guessed: the join key is **recomputed from the deck** by
  :func:`digest_from_deck` (``MAS_INP``'s ``%LPD_SHF`` body IS the payload
  ``Pattern.digest`` hashes — equal to the directory-name digest on 400/400 live
  work dirs), and ``manifest.csv``'s ``case_path`` restores the ORIGINAL
  directory name and parent, which feeds the unchanged seed-tag / ``%JOB_IDE``
  rules above.  ``key_source`` records ``deck`` or ``name`` per row.
* **``cyNN`` chains.**  ``regen/<id>/cy01..cyNN`` (and ``local_3GA``'s
  ``full_cyNN`` / ``local_eqlp_ws``'s ``cyNN``) keep every cycle.  The final cycle
  is the highest ``cyNN`` carrying both ``MAS_SUM`` and ``MAS_OUT`` — pinned to
  ``manifest.csv``'s ``cyc_max`` when the archive states one, so a sibling branch
  such as ``quarter_cy99`` cannot pose as the chain's end.  Its evidence is
  ``regen:cyNN/N`` / ``chain:cyNN/N``; every lower cycle is ``superseded``.

Cases the archive cannot adjudicate are refused, not guessed: ``ambiguous_chain``
(a multi-work-dir case with neither a staged name nor a clean ``cyNN`` layout),
``no_manifest_row``, ``not_converged``.  A single-work-dir case that
``manifest.csv`` classifies ``final_cycle_only`` — the harvest-then-purge shape
that dominates every origin — is evidenced ``manifest:final_cycle_only``;
:func:`apply`'s EFPD-vs-``cyclen`` guard is what stops a mid-chain survivor that
was mis-classified, exactly as it does for a live tree.

``apply`` contract (inherited from ``backfill_flatness``)
--------------------------------------------------------
* **Idempotent** — writes only where the column is currently null; a run with
  nothing to do writes no file.
* **Atomic + order-preserving** — patches a FRESH read by ``record_id`` through
  the store's own ``_atomic_write`` / ``frame_to_table``; row order is unchanged.
* **Never destructive** — the parquet is copied to
  ``records.parquet.bak_pre_fxy_backfill_<YYYYMMDD>`` before the write, and every
  refusal (ambiguous digest, cycle-length mismatch, broken ``F_r <= F_xy <= F_q``)
  is COUNTED, never guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from ..data.fxy import NONFINITE_SENTINEL, fxy_from_work_dir
from ..safelog import safe_logger
from ..data.geometry import to_canonical_from_shf

#: CSV header, in order.  ``key_source`` is appended (never inserted) so a CSV
#: written by an older scan still reads back through :func:`read_csv`.
CSV_COLUMNS: tuple[str, ...] = (
    "digest16", "work_dir", "cycle_evidence", "f_xy", "f_xya",
    "n_steps", "sane", "reason", "efpd_max", "key_source",
)

#: ``efpd_max`` (MAS_OUT) vs the store's ``cyclen``: same cycle or not this dir.
#: MASTER prints EFPD to 3 decimals and the store keeps the same number, so 1
#: EFPD is pure slack; a wrong cycle differs by tens.
CYCLEN_TOL_EFPD = 1.0
#: Slack on the physical ordering ``F_r <= F_xy <= F_q`` (§5.4): MAS_OUT/MAS_SUM
#: print 4 decimals, so anything beyond this is a real violation.
INEQUALITY_TOL = 1.0e-3

_DIGEST_DIR_RE = re.compile(r"^(?P<digest>[0-9a-f]{16})__(?P<tag>.+)$")
#: the ``-<contentkey10>-<mkdtemp8>`` suffix ``master.py`` appends to the case name.
_MKDTEMP_SUFFIX_RE = re.compile(r"-[0-9a-f]{10}-.{8}$")
_RESTART_REF_RE = re.compile(r"^\s*(?:\S*[\\/])?(MAS_RST\.\S+)\s*$", re.MULTILINE)
_JOB_IDE_RE = re.compile(r"%JOB_IDE[^\n]*\n[ \t]*\S+[ \t]+(\d+)")

#: ``<digest16>__`` ANYWHERE in a name — the last-resort key when the deck will
#: not parse.  The lookbehind stops a longer hex run donating its first 16 chars.
_DIGEST_ANY_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{16})__")
#: a chain cycle directory: ``cy1``, ``cy02``, ``full_cy11``, ``quarter_cy99``.
#: The prefix must be empty or end in ``_`` so a flattened case name that merely
#: happens to end in digits cannot be read as a cycle.
_CYCLE_DIR_RE = re.compile(r"^(?P<prefix>|.*_)cy(?P<n>\d{1,3})$", re.IGNORECASE)

#: the archive's per-case index (``origin``/``case_name`` -> original path).
MANIFEST_NAME = "manifest.csv"
#: per-chain sidecar a ``regen/<id>/`` chain carries.
META_CHAIN_NAME = "_meta_chain.json"
#: how far above a work dir its collected case directory may sit.
_CASE_LOOKUP_DEPTH = 4

#: ``cycle_evidence`` prefixes that mean "this dir IS the chain's final cycle".
#: ``final`` is the live-tree verdict; the archive layouts carry their evidence
#: inline (``regen:cy12/12``) so a CSV row stays self-describing.
FINAL_EVIDENCE_PREFIXES: tuple[str, ...] = ("manifest:", "chain:", "regen:")


def is_final_evidence(evidence: str) -> bool:
    """Does this ``cycle_evidence`` mean "final cycle, usable as a label"?"""
    return evidence == "final" or evidence.startswith(FINAL_EVIDENCE_PREFIXES)


def _bucket(evidence: str) -> str:
    """Report bucket for an evidence string (``regen:cy12/12`` -> ``regen``)."""
    return evidence.split(":", 1)[0]


def _int(raw: str | None) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# join key
# --------------------------------------------------------------------------- #
def digest_of_packed(packed: str) -> str:
    """``Pattern.digest`` of a store ``pattern`` cell, without rebuilding it.

    ``schema.pack_pattern`` IS ``Pattern.canonical()`` and ``Pattern.digest`` is
    ``sha256(canonical.encode("ascii"))[:16]`` (``vendor/masterrl/domain.py:186``),
    so the store column already holds the exact hashed payload — the round trip
    through :func:`~..data.schema.unpack_pattern` would parse 69 fuel cards per
    row (5.1 M for the 74 k-row store) to arrive at the same 16 hex characters.
    ``tests/test_backfill_fxy.py`` pins the two against each other.

    NOTE the work-dir prefix is this digest, **not** ``record_id[:16]``:
    ``record_id`` also hashes library / case pair / deck knobs
    (``schema.compute_record_id``), and 0 of 586 measured dirs matched it.
    """
    return sha256(packed.encode("ascii")).hexdigest()[:16]


def _deck_text(deck: Path) -> str:
    """Decode a MASTER deck on a Windows/Korean box (the ladder
    :func:`..data.fxy._read_text_flex` uses; copied so :func:`_deck_facts`'s own
    decoding — and therefore the live-tree verdicts — is left exactly as it was)."""
    data = deck.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("ascii", errors="replace")


def digest_from_deck(work_dir: str | Path) -> str | None:
    """``Pattern.digest`` recomputed from the work dir's own ``MAS_INP``.

    The archive sweep compressed each case's nested path into ONE directory name
    with a ``~`` elision, so the staged ``<digest16>__`` prefix is usually gone
    and may even be truncated mid-hash — the directory name is not a join key
    there.  The DECK is untouched: its ``%LPD_SHF`` body is exactly the payload
    ``Pattern.digest`` hashes, so re-parsing it with the store's own routines
    (:func:`..data.geometry.to_canonical_from_shf` -> :attr:`Pattern.digest`,
    the same pair ``extract_b.build_pattern_index`` uses) rebuilds the key with
    no re-implementation.  Verified equal to the directory-name digest on 400/400
    live work dirs.

    ``None`` — never an exception — for a missing/unreadable deck and for a deck
    whose ``%LPD_SHF`` does not parse (a bootstrap cy01 deck is ``irrst=0`` and
    carries no shuffle cards at all).
    """
    deck = Path(work_dir) / "MAS_INP"
    try:
        if not deck.is_file():
            return None
        text = _deck_text(deck)
    except OSError:
        return None
    try:
        return to_canonical_from_shf(text).digest
    except (ValueError, KeyError, IndexError, AssertionError):
        return None


def _join_key(work_dir: Path, name: str) -> tuple[str, str]:
    """``(digest16, key_source)`` for an ARCHIVED work dir — deck first, name last.

    The name fallback searches the recovered original name and then the on-disk
    one; ``("", "")`` when neither yields a key (the row is still emitted, with
    its evidence, so the refusal is auditable).
    """
    digest = digest_from_deck(work_dir)
    if digest:
        return digest, "deck"
    for candidate in (name, work_dir.name):
        match = _DIGEST_ANY_RE.search(candidate)
        if match:
            return match.group(1), "name"
    return "", ""


# --------------------------------------------------------------------------- #
# archive layout (manifest.csv / regen chains)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _CaseInfo:
    """What the collection sweep knows about one collected case directory.

    Empty/``None`` fields mean "the archive does not say" — never a default that
    would let an unproven dir be adjudicated final.
    """

    case_dir: Path
    #: the ORIGINAL directory name (``<digest16>__<tag>…``), from ``case_path``.
    orig_name: str = ""
    #: the ORIGINAL parent (the worker dir the ``superseded`` rule compares in).
    orig_parent: str = ""
    chain_class: str = ""
    cyc_max: int | None = None
    n_cycles: int | None = None
    #: ``manifest`` or ``regen`` — selects the ``chain:``/``regen:`` label.
    source: str = "manifest"
    #: ``_meta_chain.json``'s verdict; ``False`` refuses the whole chain.
    converged: bool | None = None


def _load_manifest(root: Path, manifest: str | Path | None = None
                   ) -> dict[tuple[str, str], _CaseInfo]:
    """``(origin, case_name) -> _CaseInfo`` from the archive's ``manifest.csv``.

    Looked up beside the scan root and one level above it, so both
    ``scan(<archive>)`` and ``scan(<archive>/srv181)`` find it; ``{}`` (i.e. a
    plain live-runs scan) when there is none.  Keyed by name pair rather than by
    path so no path-normalisation difference can silently drop the join.
    """
    candidates = ([Path(manifest)] if manifest
                  else [root / MANIFEST_NAME, root.parent / MANIFEST_NAME])
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}
    index: dict[tuple[str, str], _CaseInfo] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            origin = (row.get("origin") or "").strip()
            case_name = (row.get("case_name") or "").strip()
            if not origin or not case_name:
                continue
            original = (row.get("case_path") or "").replace("\\", "/").rstrip("/")
            head, _, tail = original.rpartition("/")
            index[(origin, case_name)] = _CaseInfo(
                case_dir=Path(case_name),
                orig_name=tail,
                orig_parent=head,
                chain_class=(row.get("chain_class") or "").strip(),
                cyc_max=_int(row.get("cyc_max")),
                n_cycles=_int(row.get("n_cycles")),
            )
    return index


def _regen_info(chain_dir: Path) -> _CaseInfo:
    """``_CaseInfo`` for a ``regen/<id>/`` chain, from its ``_meta_chain.json``.

    The sidecar carries the record's ``pattern`` and whether the chain converged;
    an unreadable one degrades to "a chain, nothing else known" rather than
    failing the scan."""
    meta: dict[str, Any] = {}
    try:
        meta = json.loads((chain_dir / META_CHAIN_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    converged = meta.get("converged")
    return _CaseInfo(
        case_dir=chain_dir, chain_class="regen", source="regen",
        n_cycles=_int(meta.get("n_cycles")),
        converged=converged if isinstance(converged, bool) else None,
    )


def _case_of(work_dir: Path, manifest: dict[tuple[str, str], _CaseInfo]
             ) -> _CaseInfo | None:
    """The collected case a work dir belongs to, or ``None`` for a live tree."""
    node = work_dir
    for _ in range(_CASE_LOOKUP_DEPTH):
        info = manifest.get((node.parent.name, node.name))
        if info is not None:
            return replace(info, case_dir=node)
        if (node / META_CHAIN_NAME).is_file():
            return _regen_info(node)
        if node.parent == node:
            break
        node = node.parent
    # A manifest was loaded, so this IS an archive scan and the dir is simply not
    # covered by it: keep it on the archive's terms (deck-derived key, an
    # explicit ``no_manifest_row``) instead of silently taking the live path.
    return _CaseInfo(case_dir=work_dir) if manifest else None


def _chain_evidence(work_dir: Path, info: _CaseInfo) -> str | None:
    """``cy<NN>/<N>`` verdict for a ``cyNN``-chain case; ``None`` if not one.

    Refuses (``None``) whenever the case also holds a work dir that is NOT a
    cycle directory — a ``bootstrap_work`` case mixes ``cy1`` with unnumbered
    ``master__bootstrap-*`` dirs, and there "the highest cyNN" is not the chain's
    end.  ``manifest.csv``'s ``cyc_max`` wins over the observed maximum when the
    archive states one, so a parallel branch (``quarter_cy99`` beside
    ``full_cy11..13``) cannot pose as the final cycle of the chain it sits in.
    """
    match = _CYCLE_DIR_RE.match(work_dir.name)
    if match is None or work_dir.parent != info.case_dir:
        return None
    prefix = match.group("prefix")
    number = int(match.group("n"))
    complete: dict[int, Path] = {}
    try:
        children = sorted(info.case_dir.iterdir())
    except OSError:                               # pragma: no cover - unreadable
        return None
    for child in children:
        if not child.is_dir():
            continue
        sub = _CYCLE_DIR_RE.match(child.name)
        if sub is None:
            if (child / "MAS_OUT").is_file():     # a non-cycle work dir: refuse
                return None
            continue
        if sub.group("prefix") != prefix:
            continue
        if (child / "MAS_OUT").is_file() and (child / "MAS_SUM").is_file():
            complete[int(sub.group("n"))] = child
    if not complete:
        return "superseded"
    target = info.cyc_max if info.cyc_max is not None else max(complete)
    if number != target or number not in complete:
        return "superseded"
    total = info.n_cycles or len(complete)
    label = "regen" if info.source == "regen" else "chain"
    return f"{label}:cy{number:02d}/{total}"


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
@dataclass
class ScanRow:
    digest16: str
    work_dir: str
    cycle_evidence: str
    f_xy: float | None = None
    f_xya: float | None = None
    n_steps: int = 0
    sane: bool = False
    reason: str = ""
    efpd_max: float | None = None
    #: ``deck`` (recomputed from ``%LPD_SHF``), ``name``, or ``""`` (no key).
    key_source: str = ""

    def as_csv(self) -> dict[str, Any]:
        row = asdict(self)
        row["sane"] = "1" if self.sane else "0"
        for key in ("f_xy", "f_xya", "efpd_max"):
            row[key] = "" if row[key] is None else f"{row[key]:.6g}"
        return row


@dataclass
class ScanReport:
    root: str
    n_dirs: int = 0
    evidence: dict[str, int] = field(default_factory=dict)
    n_sane: int = 0
    #: ``deck`` / ``name`` split over the rows that reached a final verdict.
    key_sources: dict[str, int] = field(default_factory=dict)
    #: ``manifest.csv`` rows loaded (0 = a plain live-runs scan).
    n_manifest_rows: int = 0

    def count(self, evidence: str) -> None:
        """Tally one dir under its evidence BUCKET (``regen:cy12/12`` ->
        ``regen``): the per-chain detail belongs in the CSV row, not in a report
        that would otherwise grow one line per chain."""
        bucket = _bucket(evidence)
        self.evidence[bucket] = self.evidence.get(bucket, 0) + 1

    def uncount(self, evidence: str) -> None:
        bucket = _bucket(evidence)
        self.evidence[bucket] = self.evidence.get(bucket, 0) - 1

    def count_key(self, key_source: str) -> None:
        name = key_source or "none"
        self.key_sources[name] = self.key_sources.get(name, 0) + 1

    def render(self) -> str:
        lines = [f"root             {self.root}", f"work dirs        {self.n_dirs}"]
        if self.n_manifest_rows:
            lines.append(f"manifest rows    {self.n_manifest_rows}")
        for name in sorted(self.evidence):
            lines.append(f"  {name:<14} {self.evidence[name]}")
        lines.append(f"sane f_xy        {self.n_sane}")
        for name in sorted(self.key_sources):
            lines.append(f"  key {name:<10} {self.key_sources[name]}")
        return "\n".join(lines)


def _seed_tag(dir_name: str) -> tuple[str, str] | None:
    """``(digest16, seed restart-name prefix)`` from a staged work-dir name.

    ``master.py`` truncates the case name to 40 chars before appending its own
    suffix, so the tag is a PREFIX of the seed restart's file name, not the whole
    of it (measured: ``MAS_RST.APRQ_10_0615.1`` for a seed ``…_0615.11``) — which
    is why the first-cycle test below is ``startswith``, not equality.
    """
    match = _DIGEST_DIR_RE.match(dir_name)
    if not match:
        return None
    return match.group("digest"), _MKDTEMP_SUFFIX_RE.sub("", match.group("tag"))


def _deck_facts(work_dir: Path) -> tuple[str | None, int | None]:
    """``(MAS_INP's restart reference, its %JOB_IDE cycle)``; ``(None, None)`` if
    the deck is missing or unreadable."""
    deck = work_dir / "MAS_INP"
    try:
        if not deck.is_file():
            return None, None
        text = deck.read_bytes().decode("ascii", errors="replace")
    except OSError:
        return None, None
    restart = _RESTART_REF_RE.search(text)
    job = _JOB_IDE_RE.search(text)
    return (restart.group(1) if restart else None,
            int(job.group(1)) if job else None)


def _chain_position(work_dir: Path, digest: str, tag: str,
                    cycles: dict[tuple[str, str], int], parent_key: str) -> str:
    """The staged-name rules: ``first_cycle`` / ``superseded`` / ``final``.

    ``parent_key`` is the directory the sibling comparison happens within — the
    work dir's own parent in a live tree, the ORIGINAL worker directory for a
    flattened archive case (where every case was lifted into one flat origin
    folder and would otherwise all look like siblings of each other).
    """
    restart, job = _deck_facts(work_dir)
    if restart is None:
        return "no_deck"
    if tag and restart.startswith(tag):
        return "first_cycle"
    best = cycles.get((parent_key, f"{digest}__{tag}"))
    if best is not None and job is not None and job < best:
        return "superseded"
    return "final"


def _effective_name(work_dir: Path, info: _CaseInfo | None) -> str:
    """The name the staged-layout rules read: the ORIGINAL one when the archive
    recovered it, else the directory's own.

    Only the case directory itself gets the recovered name — a ``cyNN`` child
    inside a collected case keeps its own, or every cycle of a chain would answer
    to the case's seed tag.
    """
    if info is not None and info.orig_name and work_dir == info.case_dir:
        return info.orig_name
    return work_dir.name


def _adjudicate(work_dir: Path, cycles: dict[tuple[str, str], int],
                info: _CaseInfo | None = None) -> tuple[str, str, str]:
    """``(cycle_evidence, digest16, key_source)`` — see the module docstring.

    ``info`` is ``None`` for a live runs tree, and that path is exactly the
    original one: name-derived key, name-derived seed tag, sibling comparison
    within the work dir's own parent.
    """
    if (work_dir / NONFINITE_SENTINEL).exists():
        return "nonfinite", "", ""
    if not (work_dir / "MAS_SUM").is_file():
        return "no_mas_sum", "", ""

    parsed = _seed_tag(_effective_name(work_dir, info))
    if info is None:                              # live runs tree — unchanged
        if parsed is None:
            return "no_digest", "", ""
        digest, tag = parsed
        return (_chain_position(work_dir, digest, tag, cycles,
                                str(work_dir.parent)), digest, "name")

    digest, key_source = _join_key(work_dir, _effective_name(work_dir, info))
    if info.converged is False:
        return "not_converged", digest, key_source
    chained = _chain_evidence(work_dir, info)
    if chained is not None:
        return chained, digest, key_source
    if parsed is not None:
        parent_key = info.orig_parent or str(work_dir.parent)
        return (_chain_position(work_dir, parsed[0], parsed[1], cycles,
                                parent_key), digest, key_source)
    if not digest:
        return "no_digest", "", ""
    if not info.chain_class:
        return "no_manifest_row", digest, key_source
    # Single surviving work dir + the archive's "only the final cycle was left
    # behind" classification.  Everything else is a chain we could not position
    # this dir inside -- counted, never guessed.
    siblings = sum(1 for p in info.case_dir.rglob("MAS_OUT"))
    if siblings == 1 and info.chain_class == "final_cycle_only":
        return "manifest:final_cycle_only", digest, key_source
    return "ambiguous_chain", digest, key_source


def scan(root: str | Path, *, manifest: str | Path | None = None,
         log: Callable[[str], None] | None = None
         ) -> tuple[list[ScanRow], ScanReport]:
    """Adjudicate every MASTER work dir under ``root`` and parse the finals.

    ``manifest`` overrides the ``manifest.csv`` auto-discovery (beside ``root``
    and one level above it); a tree with neither it nor a ``_meta_chain.json`` is
    scanned exactly as before.
    """
    # Encoding-safe default logger: a redirected Windows stdout is cp949 and a
    # single em-dash used to raise UnicodeEncodeError mid-run (2026-08-30).
    log = safe_logger(log or (lambda m: print(m, flush=True)))
    root = Path(root)
    dirs = sorted({p.parent for p in root.rglob("MAS_OUT")})
    cases = _load_manifest(root, manifest)
    report = ScanReport(root=str(root), n_dirs=len(dirs),
                        n_manifest_rows=len(cases))

    # Highest %JOB_IDE per (parent dir, "<digest16>__<tag>") chain — the sibling
    # comparison the ``superseded`` rule needs.  One deck read per dir, ~8 kB.
    infos: dict[Path, _CaseInfo | None] = {}
    cycles: dict[tuple[str, str], int] = {}
    for work_dir in dirs:
        info = _case_of(work_dir, cases)
        infos[work_dir] = info
        parsed = _seed_tag(_effective_name(work_dir, info))
        if parsed is None:
            continue
        _restart, job = _deck_facts(work_dir)
        if job is None:
            continue
        parent_key = (info.orig_parent if info is not None and info.orig_parent
                      else str(work_dir.parent))
        key = (parent_key, f"{parsed[0]}__{parsed[1]}")
        cycles[key] = max(cycles.get(key, job), job)

    rows: list[ScanRow] = []
    for work_dir in dirs:
        evidence, digest, key_source = _adjudicate(work_dir, cycles,
                                                   infos[work_dir])
        report.count(evidence)
        row = ScanRow(digest16=digest, work_dir=str(work_dir),
                      cycle_evidence=evidence, key_source=key_source)
        if is_final_evidence(evidence):
            peaks = fxy_from_work_dir(work_dir)
            if peaks is None:
                report.uncount(evidence)
                row.cycle_evidence = "unreadable"
                report.count("unreadable")
            else:
                report.count_key(key_source)
                row.f_xy, row.f_xya = peaks.f_xy, peaks.f_xya
                row.n_steps, row.sane, row.reason = (
                    peaks.n_steps, peaks.sane, peaks.reason)
                row.efpd_max = peaks.efpd_max
                if peaks.sane:
                    report.n_sane += 1
        rows.append(row)
    log(f"[backfill_fxy] scanned {len(dirs)} work dir(s) under {root}")
    return rows, report


def write_csv(rows: list[ScanRow], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv())
    return out


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
@dataclass
class ApplyReport:
    store_dir: str
    csv_path: str
    n_csv_rows: int = 0
    n_final: int = 0             # csv rows adjudicated final AND sane
    n_digests: int = 0           # distinct usable digests
    n_dup_digest: int = 0        # one digest, two dirs disagreeing on f_xy
    n_no_store_row: int = 0      # digest not in the store
    n_ambiguous: int = 0         # digest hits >1 record_id
    n_cycle_mismatch: int = 0    # efpd_max != store cyclen
    n_inequality: int = 0        # F_r <= F_xy <= F_q violated
    n_already: int = 0           # store cell already populated
    n_populated: int = 0
    #: ``deck`` / ``name`` split over the final+sane CSV rows (archive scans).
    key_sources: dict[str, int] = field(default_factory=dict)
    per_campaign: dict[str, int] = field(default_factory=dict)
    backup: str = ""
    dry_run: bool = False
    wrote: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"store              {self.store_dir}",
            f"csv                {self.csv_path}  ({self.n_csv_rows} row(s))",
            f"final & sane       {self.n_final}  -> {self.n_digests} digest(s)",
            *(f"    key {k:<12} {v}" for k, v in sorted(self.key_sources.items())),
            f"  dup digest       {self.n_dup_digest}",
            f"  no store row     {self.n_no_store_row}",
            f"  ambiguous        {self.n_ambiguous}",
            f"  cycle mismatch   {self.n_cycle_mismatch}",
            f"  F_r<=F_xy<=F_q   {self.n_inequality}",
            f"  already filled   {self.n_already}",
            f"populated          {self.n_populated}",
        ]
        for campaign in sorted(self.per_campaign):
            lines.append(f"    {campaign:<28} {self.per_campaign[campaign]}")
        lines.append(f"backup             {self.backup or '-'}")
        lines.append(
            f"wrote              {self.wrote}{'  (dry-run)' if self.dry_run else ''}")
        return "\n".join(lines)


def _float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def apply(csv_path: str | Path, store_dir: str | Path = "data/store", *,
          dry_run: bool = False, log: Callable[[str], None] | None = None
          ) -> ApplyReport:
    """Join a :func:`scan` CSV to ``records.parquet`` and fill the null cells."""
    import pandas as pd
    import pyarrow.parquet as pq

    from ..data.store import (
        RECORDS_NAME, _atomic_write, ensure_schema_columns, frame_to_table,
    )

    # Encoding-safe default logger: a redirected Windows stdout is cp949 and a
    # single em-dash used to raise UnicodeEncodeError mid-run (2026-08-30).
    log = safe_logger(log or (lambda m: print(m, flush=True)))
    rep = ApplyReport(store_dir=str(store_dir), csv_path=str(csv_path),
                      dry_run=dry_run)

    # -- CSV -> {digest: (f_xy, f_xya, efpd_max)} --------------------------- #
    collected: dict[str, list[tuple[float, float | None, float | None]]] = {}
    for row in read_csv(csv_path):
        rep.n_csv_rows += 1
        if not is_final_evidence(row.get("cycle_evidence") or "") \
                or row.get("sane") != "1":
            continue
        f_xy = _float(row.get("f_xy", ""))
        digest = (row.get("digest16") or "").strip()
        if f_xy is None or not digest:
            continue
        rep.n_final += 1
        rep.key_sources[row.get("key_source") or "name"] = (
            rep.key_sources.get(row.get("key_source") or "name", 0) + 1)
        collected.setdefault(digest, []).append(
            (f_xy, _float(row.get("f_xya", "")), _float(row.get("efpd_max", ""))))

    peaks: dict[str, tuple[float, float | None, float | None]] = {}
    for digest, entries in collected.items():
        # Two retained finals for one pattern (a re-run against a different
        # restart).  Agreeing to the printed precision is one measurement;
        # disagreeing means one of them is not the final cycle -- never guess.
        if max(e[0] for e in entries) - min(e[0] for e in entries) > 1.0e-6:
            rep.n_dup_digest += 1
            continue
        peaks[digest] = entries[0]
    rep.n_digests = len(peaks)

    # -- store side --------------------------------------------------------- #
    path = Path(store_dir) / RECORDS_NAME
    df = ensure_schema_columns(pd.read_parquet(path))
    digests = df["pattern"].astype(str).map(digest_of_packed)

    by_digest: dict[str, list[int]] = {}
    for pos, digest in enumerate(digests.to_numpy(dtype=object)):
        if digest in peaks:
            by_digest.setdefault(str(digest), []).append(pos)

    rids = df["record_id"].astype(str).to_numpy(dtype=object)
    campaigns = (df["campaign"].astype(str).to_numpy(dtype=object)
                 if "campaign" in df.columns else [""] * len(df))
    stored_fxy = pd.to_numeric(df["f_xy"], errors="coerce").to_numpy()
    cyclen = pd.to_numeric(df["cyclen"], errors="coerce").to_numpy()
    f_r = pd.to_numeric(df["f_r"], errors="coerce").to_numpy()
    f_q = pd.to_numeric(df["f_q"], errors="coerce").to_numpy()

    new_fxy: dict[str, float] = {}
    new_fxya: dict[str, float] = {}
    for digest, (value, value_a, efpd) in peaks.items():
        positions = by_digest.get(digest)
        if not positions:
            rep.n_no_store_row += 1
            continue
        if len({rids[p] for p in positions}) > 1:
            # Pattern-only key; two records can share it (different library /
            # case pair).  Design §3.3: count, never guess.
            rep.n_ambiguous += 1
            continue
        pos = positions[0]
        if efpd is not None and cyclen[pos] == cyclen[pos] and \
                abs(float(cyclen[pos]) - efpd) > CYCLEN_TOL_EFPD:
            rep.n_cycle_mismatch += 1
            continue
        if (f_r[pos] == f_r[pos] and float(f_r[pos]) > value + INEQUALITY_TOL) or \
                (f_q[pos] == f_q[pos] and value > float(f_q[pos]) + INEQUALITY_TOL):
            rep.n_inequality += 1
            continue
        # ANY copy of this record_id already carrying a value counts as filled:
        # the store can hold duplicate rows per record_id, and half-filling them
        # would leave one record with two different f_xy values on disk.
        if any(stored_fxy[p] == stored_fxy[p] for p in positions):
            rep.n_already += 1
            continue
        new_fxy[str(rids[pos])] = value
        if value_a is not None:
            new_fxya[str(rids[pos])] = value_a
        rep.n_populated += 1
        name = str(campaigns[pos])
        rep.per_campaign[name] = rep.per_campaign.get(name, 0) + 1

    if not new_fxy:
        log("[backfill_fxy] nothing to populate; store untouched")
        return rep
    log(f"[backfill_fxy] {rep.n_populated} row(s) to populate")
    if dry_run:
        return rep

    # Backup BEFORE the write (the flatness backfill had maps.npz to recompute
    # from; this one's source tree can be deleted or on another box).
    backup = path.with_name(f"{path.name}.bak_pre_fxy_backfill_"
                            f"{date.today():%Y%m%d}")
    shutil.copy2(path, backup)
    rep.backup = str(backup)

    # Fresh read + patch by record_id: row order survives and any row appended
    # since the scan is carried through untouched.
    current = ensure_schema_columns(pd.read_parquet(path))
    rid_col = current["record_id"].astype(str)
    for column, values in (("f_xy", new_fxy), ("f_xya", new_fxya)):
        # NULL-ONLY fill, stored value wins.  The store can hold more than one
        # row per ``record_id`` (readers de-duplicate), and ``Series.map`` hits
        # every one of them — so the mapped value must lose to anything already
        # there, or a duplicate that was already labelled gets overwritten.
        # (``backfill_flatness`` takes the opposite priority on purpose: it exists
        # to CORRECT stale scalars; this pass only fills gaps.)
        stored = pd.to_numeric(current[column], errors="coerce")
        current[column] = stored.where(stored.notna(), rid_col.map(values))
    table = frame_to_table(current)
    _atomic_write(path, lambda p: pq.write_table(table, p))
    rep.wrote = True
    log(f"[backfill_fxy] wrote {rep.n_populated} value(s); backup at {backup.name}")
    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m lpopt.tools.backfill_fxy",
        description="retro-label f_xy / f_xya from MAS_OUT files on disk")
    sub = ap.add_subparsers(dest="mode", required=True)

    scan_ap = sub.add_parser("scan", help="walk a runs tree; emit a CSV (no store)")
    scan_ap.add_argument("--root", required=True)
    scan_ap.add_argument("--out", required=True)
    scan_ap.add_argument(
        "--manifest", default=None,
        help="archive manifest.csv (default: auto-discovered beside --root "
             "or one level above it; absent for a live runs tree)")

    apply_ap = sub.add_parser("apply", help="join a scan CSV into records.parquet")
    apply_ap.add_argument("--csv", required=True)
    apply_ap.add_argument("--store-dir", default="data/store")
    apply_ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    if args.mode == "scan":
        rows, rep = scan(args.root, manifest=args.manifest)
        write_csv(rows, args.out)
        print(rep.render())
        print(f"csv              {args.out}")
        return 0
    rep = apply(args.csv, args.store_dir, dry_run=args.dry_run)
    print(rep.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CSV_COLUMNS",
    "CYCLEN_TOL_EFPD",
    "FINAL_EVIDENCE_PREFIXES",
    "INEQUALITY_TOL",
    "MANIFEST_NAME",
    "META_CHAIN_NAME",
    "ApplyReport",
    "ScanReport",
    "ScanRow",
    "apply",
    "digest_from_deck",
    "digest_of_packed",
    "is_final_evidence",
    "main",
    "read_csv",
    "scan",
    "write_csv",
]
