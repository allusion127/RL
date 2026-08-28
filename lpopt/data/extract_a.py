"""Dataset A extraction pipeline (plan 4.2, milestone M2-A).

Stages (each a function; orchestrated by :func:`run_extract_a`):

  a. **cache scan** — 11 ``sa_2b_cache*.jsonl`` files across three workspaces.
     Line 1 of each file is a fingerprint header (not a record).  Records are
     deduplicated across files by a rotation-normalized, sorted canonical key.
  b. **library resolution** — per record, via the ``run_meta.json`` of the run
     that produced it (authoritative), falling back to fresh-type-name pattern
     rules; unresolved records are tagged ``library_id="unresolved:<names>"`` and
     counted rather than guessed.
  c. **metric mapping** — cache ``rec.metrics`` -> schema target columns.
  d. **case-dir join + harvest** — for each ``runs/*/cases/*`` dir take the final
     ``cyNN``, parse ``MAS_INP %LPD_SHF`` -> canonical key -> match, then parse
     ``MAS_SUM`` -> ``cbc_max`` + EDIT5 maps (ProcessPoolExecutor).
  e. **derived columns** — pattern, feed, n_batches/depth2 (GeneralOrbitGenome),
     e_core/e_split (FuelLibrary), sym_class, provenance.
  f. **write** — :class:`StoreWriter` + ``data/reports/extract_report.md``.

Windows/Korean paths: pathlib throughout, utf-8-sig with cp949 fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..search.genome import GeneralOrbitGenome, GenomeError
from .edit5 import cbc_max, parse_mas_sum, stack_maps
from .flatness import record_flatness
from .fuel_types import FuelLibrary, core_enrichment_split
from .geometry import to_cache_key, to_canonical_from_cache_key, to_canonical_from_shf
from .schema import (
    MOCHA_DECK_KNOBS,
    SYM_CLASS,
    CanonicalRecord,
    compute_record_id,
)
from .store import StoreWriter

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
_CY_RE = re.compile(r"^cy(\d+)$")
_FRESH_SPEC_RE = re.compile(r"^F:(?P<batch>[^r]+?)(?:r(?P<rot>\d+))?$")
_RUN_RE = re.compile(r"runs[\\/]([^\\/]+)[\\/]")
#: single-zero-padded fresh batch label (``A04``, ``B01``) -> bare (``A4``, ``B1``).
_FRESH_PAD_RE = re.compile(r"^([A-Za-z]+)0(\d)$")
_DECK_KNOBS = MOCHA_DECK_KNOBS

#: run_meta source-path -> library markers.
_LIB_260624 = ("\\260624\\", "/260624/")
_LIB_5851 = ("IGD_16", "5.8_5.1")


# --------------------------------------------------------------------------- #
# text reader (Windows/Korean cp949 fallback)
# --------------------------------------------------------------------------- #
def _read_text_flex(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("ascii", errors="replace")


# --------------------------------------------------------------------------- #
# canonical dedup key (shared by scan + harvest join)
# --------------------------------------------------------------------------- #
def _unpad_fresh_batch(batch: str) -> str:
    """Strip a SINGLE zero-pad from a fresh batch label: ``A04`` -> ``A4``,
    ``B01`` -> ``B1``; a no-op for already-bare (``A4``) or multi-digit (``A10``)
    names.  Never merges distinct physical types — ``A04`` and ``A4`` are the SAME
    fuel type (the roster resolves them interchangeably, see :func:`_type_resolver`).
    """
    m = _FRESH_PAD_RE.match(batch)
    return f"{m.group(1)}{m.group(2)}" if m else batch


def _normalize_spec(spec: str) -> str:
    """Normalize a cache-key spec: add the ``r0`` suffix to bare fresh specs AND
    strip single-zero fresh-name padding.

    Applied SYMMETRICALLY by :func:`dedup_key_of` to both the sa_2b cache key and
    the per-cycle ``MAS_INP %LPD_SHF``-derived key.  The cache writes fresh types
    zero-padded (``F:A04``) while the deck writes them bare (``F:A4``); normalizing
    the padding on both sides is what lets a cache candidate join its surviving
    case dir — without it the 5.8_5.1 corpus never matched a case dir and its
    ``cbc_max`` was masked to ``boc_only`` (forensic 20260720).  260624 (``C##``)
    matching is unchanged: the normalization is symmetric and idempotent on the
    already-consistent labels those decks use.
    """
    m = _FRESH_SPEC_RE.match(spec)
    return f"F:{_unpad_fresh_batch(m.group('batch'))}r0" if m else spec


def dedup_key_of(key: Iterable[Iterable[Any]]) -> tuple[tuple[int, int, str], ...]:
    """Rotation-normalized, sorted canonical dedup key of a rot61 cache key."""
    return tuple(sorted((int(qi), int(qj), _normalize_spec(str(spec))) for qi, qj, spec in key))


def _run_id_of(rec: dict) -> str | None:
    for field in ("eq_restart", "seed_restart"):
        path = rec.get(field) or ""
        m = _RUN_RE.search(path)
        if m:
            return m.group(1)
    return None


def _fresh_names(key: Iterable[Iterable[Any]]) -> set[str]:
    names: set[str] = set()
    for _qi, _qj, spec in key:
        s = str(spec)
        if s.startswith("F:"):
            m = _FRESH_SPEC_RE.match(s)
            if m:
                names.add(m.group("batch"))
    return names


# --------------------------------------------------------------------------- #
# workspace / cache discovery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Workspace:
    name: str
    root: Path
    is_legacy: bool

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    def caches(self) -> list[Path]:
        main = self.root / "sa_2b_cache.jsonl"
        found = [main] if main.exists() else []
        found += sorted(self.root.glob("sa_2b_cache.stale-*.jsonl"))
        return found


def resolve_workspaces(cfg: Any, deck_dir: Path) -> list[Workspace]:
    """Build :class:`Workspace` objects from ``[extract].workspaces`` paths."""
    workspaces: list[Workspace] = []
    for raw in cfg.extract.workspaces:
        p = Path(raw)
        root = p if p.is_absolute() else (deck_dir / p)
        root = root.resolve()
        is_legacy = "eqlp_ws" in str(root).replace("\\", "/").lower()
        workspaces.append(Workspace(name=root.name, root=root, is_legacy=is_legacy))
    return workspaces


# --------------------------------------------------------------------------- #
# stage (a): cache scan + dedup
# --------------------------------------------------------------------------- #
@dataclass
class RawRecord:
    dedup_key: tuple[tuple[int, int, str], ...]
    key: list[list[Any]]
    metrics: dict[str, float]
    converged: bool
    valid: bool
    failure: str
    tag: str
    fresh_names: frozenset[str]
    campaign: str
    workspace: str
    is_legacy: bool
    run_id: str | None


@dataclass
class FileStat:
    name: str
    status: str
    lines: int
    records: int
    new_unique: int


def scan_caches(workspaces: list[Workspace], *, limit: int | None = None,
                progress: bool = True) -> tuple[list[RawRecord], list[FileStat]]:
    """Read + dedup every cache file.  First occurrence wins (provenance)."""
    seen: set[tuple] = set()
    out: list[RawRecord] = []
    stats: list[FileStat] = []
    for ws in workspaces:
        for cache in ws.caches():
            disp = f"{ws.name}/{cache.name}"
            if not cache.exists():
                stats.append(FileStat(disp, "MISSING", 0, 0, 0))
                continue
            lines = records = new_unique = 0
            saw_header = False
            # Workspace-qualified so the two "sa_2b_cache" files (0_Case vs
            # eqlp_ws) do not collide in the provenance column.
            stem = f"{ws.name}:{cache.name[: -len('.jsonl')]}"
            with open(cache, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    lines += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "key" not in obj:
                        saw_header = True          # fingerprint header line
                        continue
                    records += 1
                    key = obj["key"]
                    dk = dedup_key_of(key)
                    if dk in seen:
                        continue
                    seen.add(dk)
                    new_unique += 1
                    rec = obj.get("rec", {})
                    out.append(RawRecord(
                        dedup_key=dk,
                        key=key,
                        metrics=rec.get("metrics", {}) or {},
                        converged=bool(rec.get("converged", False)),
                        valid=bool(rec.get("valid", False)),
                        failure=str(rec.get("failure", "") or ""),
                        tag=str(rec.get("tag", "") or ""),
                        fresh_names=frozenset(_fresh_names(key)),
                        campaign=stem,
                        workspace=ws.name,
                        is_legacy=ws.is_legacy,
                        run_id=_run_id_of(rec),
                    ))
                    if limit is not None and len(out) >= limit:
                        break
            status = "OK" if saw_header else "NOHDR"
            stats.append(FileStat(disp, status, lines, records, new_unique))
            if progress:
                print(f"  scan {cache.name[:44]:44s} lines={lines:6d} rec={records:6d} "
                      f"new={new_unique:6d} (unique so far {len(seen)})")
            if limit is not None and len(out) >= limit:
                return out, stats
    return out, stats


# --------------------------------------------------------------------------- #
# stage (b): library resolution
# --------------------------------------------------------------------------- #
def _library_from_meta(meta: dict) -> str | None:
    for v in (meta.get("fuel_data") or {}).values():
        if isinstance(v, dict) and v.get("source_out"):
            s = str(v["source_out"])
            if any(marker in s for marker in _LIB_260624):
                return "260624"
            if any(marker in s for marker in _LIB_5851):
                return "5.8_5.1"
    if set(meta.get("fuel_types", [])) <= {"A0", "A1"}:
        return "legacy_a"
    return None


def build_run_library_index(workspaces: list[Workspace]) -> dict[str, str]:
    """``run_id -> library_id`` from every readable ``runs/*/run_meta.json``."""
    index: dict[str, str] = {}
    for ws in workspaces:
        if not ws.runs_root.is_dir():
            continue
        for rm in ws.runs_root.glob("*/run_meta.json"):
            try:
                meta = json.loads(_read_text_flex(rm))
            except (OSError, json.JSONDecodeError):
                continue
            lib = _library_from_meta(meta)
            if lib:
                index[rm.parent.name] = lib
    return index


def _name_pattern_library(names: frozenset[str], is_legacy: bool) -> str:
    if is_legacy:
        return "legacy_a"
    if names and all(re.match(r"^A[01]$", n) for n in names):
        return "legacy_a"
    if any(re.match(r"^C\d+$", n) for n in names):        # C family -> 260624
        return "260624"
    if names and all(re.match(r"^[BC]\d$", n) for n in names):     # single B/C
        return "260624"
    if names and all(re.match(r"^[AB]0\d$", n) for n in names):    # A0#/B0#
        return "5.8_5.1"
    return "unresolved:" + ",".join(sorted(names))


def resolve_library(rec: RawRecord, run_index: dict[str, str]) -> tuple[str, str]:
    """Return ``(library_id, resolved_via)`` for a record.

    ``resolved_via`` is ``"run_meta"`` (authoritative) or ``"name_pattern"``.
    """
    if rec.run_id and rec.run_id in run_index:
        return run_index[rec.run_id], "run_meta"
    return _name_pattern_library(rec.fresh_names, rec.is_legacy), "name_pattern"


# --------------------------------------------------------------------------- #
# fuel-type name resolution (zero-padding differs by library)
# --------------------------------------------------------------------------- #
def _type_resolver(fuel_lib: FuelLibrary):
    rosters: dict[str, set[str]] = {}

    def resolve(library_id: str, raw: str) -> str | None:
        if library_id not in rosters:
            try:
                rosters[library_id] = set(fuel_lib.types(library_id))
            except Exception:
                rosters[library_id] = set()
        roster = rosters[library_id]
        if raw in roster:
            return raw
        stripped = re.sub(r"^([A-Za-z]+)0(\d)$", r"\1\2", raw)   # C01 -> C1
        if stripped in roster:
            return stripped
        padded = re.sub(r"^([A-Za-z]+)(\d)$", r"\g<1>0\2", raw)  # C1 -> C01
        if padded in roster:
            return padded
        return None

    return resolve


def _e_core_split(fuel_lib: FuelLibrary, library_id: str, batch_feed: dict[str, int],
                  resolve_type) -> tuple[float | None, float | None]:
    """Mass-weighted core-average enrichment + enrichment spread of the feed.

    Thin wrapper over :func:`fuel_types.core_enrichment_split` (the single recipe
    shared with inference featurization); passes the cached roster ``resolve_type``
    so the extraction hot loop keeps its per-library roster cache.
    """
    return core_enrichment_split(
        fuel_lib, library_id, batch_feed, resolve_type=resolve_type
    )


# --------------------------------------------------------------------------- #
# stage (c): metric mapping
# --------------------------------------------------------------------------- #
def _f(metrics: dict, key: str) -> float | None:
    v = metrics.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def map_metrics(metrics: dict) -> dict[str, float | None]:
    """Cache ``rec.metrics`` -> schema target fields (cbc_max left None)."""
    return {
        "f_r": _f(metrics, "max_frp"),
        "f_q": _f(metrics, "max_fqp"),
        "cbc_boc": _f(metrics, "boc_ppm"),
        "cyclen": _f(metrics, "cycle_length_efpd"),
        "ao_abs": _f(metrics, "max_abs_ao"),
        "cycle_burnup": _f(metrics, "cycle_burnup"),
        "discharge_burnup": _f(metrics, "discharge_burnup"),
        "max_assembly_burnup": _f(metrics, "max_assembly_burnup"),
        "max_pin_burnup": _f(metrics, "max_pin_burnup"),
        "eoc_ppm": _f(metrics, "eoc_ppm"),
        "delta_efpd": _f(metrics, "delta_efpd"),
        "n_cycles": _f(metrics, "n_cycles"),
    }


# --------------------------------------------------------------------------- #
# stage (d): case-dir join + harvest (ProcessPool worker)
# --------------------------------------------------------------------------- #
#: Target dedup-key digests, seeded per worker so non-matching dirs skip the
#: (expensive) MAS_SUM parse.  ``None`` means "harvest every dir".
_TARGET_HASHES: set[bytes] | None = None


def _key_digest(dk: tuple) -> bytes:
    """Stable cross-process digest of a dedup key (PYTHONHASHSEED-independent)."""
    return hashlib.blake2b(repr(dk).encode("utf-8"), digest_size=16).digest()


def _init_worker(target_hashes: set[bytes] | None) -> None:
    global _TARGET_HASHES
    _TARGET_HASHES = target_hashes


def _harvest_case_dir(case_dir_str: str):
    """Worker: final-cy ``MAS_INP`` -> dedup key, ``MAS_SUM`` -> cbc_max + maps.

    Returns ``(status, dedup_key|None, cbc_max|None, map_stack|None)`` where
    status is ``ok`` | ``no_sum`` | ``inp_err`` | ``sum_err`` | ``no_cy`` |
    ``nomatch``.  ``MAS_SUM`` is only parsed for dirs whose pattern matches a
    target record (see :data:`_TARGET_HASHES`).
    """
    case_dir = Path(case_dir_str)
    try:
        cys = [(int(m.group(1)), d) for d in case_dir.iterdir()
               if d.is_dir() and (m := _CY_RE.match(d.name))]
    except OSError:
        return ("no_cy", None, None, None)
    if not cys:
        return ("no_cy", None, None, None)
    final = max(cys, key=lambda t: t[0])[1]

    try:
        pat = to_canonical_from_shf(_read_text_flex(final / "MAS_INP"))
        dk = dedup_key_of(to_cache_key(pat))
    except (OSError, ValueError, KeyError, AssertionError):
        return ("inp_err", None, None, None)

    if _TARGET_HASHES is not None and _key_digest(dk) not in _TARGET_HASHES:
        return ("nomatch", None, None, None)

    msum = final / "MAS_SUM"
    if not msum.exists():
        return ("no_sum", dk, None, None)
    try:
        summary = parse_mas_sum(msum)
        cmax = float(cbc_max(summary))
        stack = stack_maps(summary).astype(np.float16)
    except (OSError, ValueError, KeyError, IndexError):
        return ("sum_err", dk, None, None)
    return ("ok", dk, cmax, stack)


def index_case_dirs(workspaces: list[Workspace], *, limit: int | None = None) -> list[Path]:
    """Every ``runs/*/cases/*`` directory across the workspaces."""
    dirs: list[Path] = []
    for ws in workspaces:
        if not ws.runs_root.is_dir():
            continue
        for cdir in ws.runs_root.glob("*/cases/*"):
            if cdir.is_dir():
                dirs.append(cdir)
                if limit is not None and len(dirs) >= limit:
                    return dirs
    return dirs


def harvest(case_dirs: list[Path], target_keys: set[tuple], *, workers: int = 8,
            progress: bool = True) -> tuple[dict[tuple, tuple[float, np.ndarray]], dict[str, int]]:
    """Harvest cbc_max + maps for the dirs whose pattern matches a record.

    Streams worker results (never accumulates file contents); keeps only maps for
    matched dedup keys.  Returns ``(harvest_by_key, status_counts)``.
    """
    harvested: dict[tuple, tuple[float, np.ndarray]] = {}
    counts: dict[str, int] = {}
    n = len(case_dirs)
    target_hashes = {_key_digest(dk) for dk in target_keys}

    def _consume(result) -> None:
        status, dk, cmax, stack = result
        counts[status] = counts.get(status, 0) + 1
        if status == "ok" and dk is not None:
            harvested[dk] = (cmax, stack)

    if workers <= 1:
        _init_worker(target_hashes)
        for i, cdir in enumerate(case_dirs, 1):
            _consume(_harvest_case_dir(str(cdir)))
            if progress and i % 2000 == 0:
                print(f"  harvest {i}/{n} dirs  matched={len(harvested)}")
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(target_hashes,)
        ) as ex:
            for i, result in enumerate(
                ex.map(_harvest_case_dir, [str(c) for c in case_dirs], chunksize=64), 1
            ):
                _consume(result)
                if progress and i % 2000 == 0:
                    print(f"  harvest {i}/{n} dirs  matched={len(harvested)}")
    return harvested, counts


# --------------------------------------------------------------------------- #
# stage (e): build canonical records
# --------------------------------------------------------------------------- #
@dataclass
class BuildStats:
    n_records: int = 0
    library_counts: dict[str, int] = None            # type: ignore[assignment]
    resolved_via: dict[str, int] = None              # type: ignore[assignment]
    unresolved: int = 0
    genome_failures: int = 0
    e_core_missing: int = 0
    converged: int = 0
    nonconverged: int = 0
    cbc_max_filled: int = 0
    footprints: int = 0
    pair_counts: dict[str, int] = None               # type: ignore[assignment]
    feed_ecore_hist: dict[tuple[int, str], int] = None  # type: ignore[assignment]
    library_cbc_filled: dict[str, int] = None        # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.library_counts = {}
        self.resolved_via = {}
        self.pair_counts = {}
        self.feed_ecore_hist = {}
        self.library_cbc_filled = {}


def _ecore_bin(e_core: float | None) -> str:
    if e_core is None or np.isnan(e_core):
        return "nan"
    lo = np.floor(e_core * 10.0) / 10.0
    return f"{lo:.1f}-{lo + 0.1:.1f}"


def build_records(raw: list[RawRecord], run_index: dict[str, str], fuel_lib: FuelLibrary,
                  harvested: dict[tuple, tuple[float, np.ndarray]], *,
                  progress: bool = True) -> tuple[list[CanonicalRecord], dict[str, np.ndarray], BuildStats]:
    """Assemble :class:`CanonicalRecord`s + map stacks (stages b, c, e)."""
    resolve_type = _type_resolver(fuel_lib)
    records: list[CanonicalRecord] = []
    maps_out: dict[str, np.ndarray] = {}
    stats = BuildStats()
    footprints: set[frozenset[int]] = set()

    for i, rr in enumerate(raw, 1):
        pattern = to_canonical_from_cache_key(rr.key)
        canonical = pattern.canonical()
        feed = pattern.feed

        # library (stage b)
        library_id, via = resolve_library(rr, run_index)
        stats.resolved_via[via] = stats.resolved_via.get(via, 0) + 1
        if library_id.startswith("unresolved:"):
            stats.unresolved += 1

        # pair + fresh footprint
        batch_feed = pattern.batch_feed()
        case_pair = "_".join(sorted(batch_feed))
        footprint = frozenset(
            idx for idx, item in enumerate(pattern.items) if item.is_fresh
        )
        footprints.add(footprint)

        # n_batches / depth2 via genome (stage e)
        try:
            genome = GeneralOrbitGenome.from_pattern(pattern)
            depth2 = int(genome.depth2_edge_count)
            n_batches = 2 if depth2 == 0 else 3
        except (GenomeError, ValueError):
            stats.genome_failures += 1
            depth2, n_batches = 0, 2

        # e_core / e_split via FuelLibrary (NaN for unresolved libraries)
        if library_id.startswith("unresolved:"):
            e_core = e_split = None
        else:
            e_core, e_split = _e_core_split(fuel_lib, library_id, batch_feed, resolve_type)
            if e_core is None:
                stats.e_core_missing += 1

        # record id + harvested cbc_max / maps (stages c, d)
        record_id = compute_record_id(canonical, library_id, case_pair, _DECK_KNOBS)
        hv = harvested.get(rr.dedup_key)
        node_peak = map_cov = None
        if hv is not None:
            cbc_max_val: float | None = hv[0]
            cbc_kind = "max"
            maps_out[record_id] = hv[1]
            maps_key: str | None = record_id
            # Same map, same pass: the flatness scalars are written with the row
            # that owns the map (never raises; a bad map leaves both null).
            node_peak, map_cov = record_flatness(hv[1])
            stats.cbc_max_filled += 1
            stats.library_cbc_filled[library_id] = (
                stats.library_cbc_filled.get(library_id, 0) + 1
            )
        else:
            cbc_max_val = None
            cbc_kind = "boc_only"
            maps_key = None

        m = map_metrics(rr.metrics)
        rec = CanonicalRecord(
            record_id=record_id,
            dataset="A",
            campaign=rr.campaign,
            stratum=None,
            generator=None,
            parent_record_id=None,
            case_pair=case_pair,
            feed=feed,
            n_batches=n_batches,
            depth2_edges=depth2,
            e_core=e_core,
            e_split=e_split,
            library_id=library_id,
            sym_class=SYM_CLASS,
            pattern=canonical,
            f_r=m["f_r"],
            f_q=m["f_q"],
            cbc_max=cbc_max_val,
            cbc_boc=m["cbc_boc"],
            cbc_kind=cbc_kind,
            cyclen=m["cyclen"],
            ao_abs=m["ao_abs"],
            cycle_burnup=m["cycle_burnup"],
            discharge_burnup=m["discharge_burnup"],
            max_assembly_burnup=m["max_assembly_burnup"],
            max_pin_burnup=m["max_pin_burnup"],
            eoc_ppm=m["eoc_ppm"],
            delta_efpd=m["delta_efpd"],
            n_cycles=m["n_cycles"],
            converged=rr.converged,
            converged_at_cap=False,
            tolerance_margin=None,
            restart_provenance="mocha_native",
            valid=rr.valid,
            failure=rr.failure,
            maps_key=maps_key,
            node_peak=node_peak,
            map_cov=map_cov,
        )
        records.append(rec)

        stats.library_counts[library_id] = stats.library_counts.get(library_id, 0) + 1
        stats.pair_counts[case_pair] = stats.pair_counts.get(case_pair, 0) + 1
        if rr.converged:
            stats.converged += 1
        else:
            stats.nonconverged += 1
        binkey = (feed, _ecore_bin(e_core))
        stats.feed_ecore_hist[binkey] = stats.feed_ecore_hist.get(binkey, 0) + 1

        if progress and i % 5000 == 0:
            print(f"  build {i}/{len(raw)} records  (maps {len(maps_out)})")

    stats.n_records = len(records)
    stats.footprints = len(footprints)
    return records, maps_out, stats


# --------------------------------------------------------------------------- #
# stage (f): report
# --------------------------------------------------------------------------- #
def write_report(path: Path, file_stats: list[FileStat], build: BuildStats,
                 harvest_counts: dict[str, int], n_case_dirs: int,
                 wall_s: float, store_dir: Path) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Dataset A extraction report\n")
    a(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- wall time: {wall_s:.1f} s")
    a(f"- store: `{store_dir}`")
    a(f"- unique records: **{build.n_records}**  (audit ground truth: 38,854)")
    a(f"- converged: {build.converged}   non-converged: {build.nonconverged}")
    a(f"- fresh-footprint uniqueness (ESS proxy): **{build.footprints}** "
      f"(audit expects ~14,682)")
    a(f"- unresolved libraries: {build.unresolved}   "
      f"genome parse failures: {build.genome_failures}   "
      f"e_core missing: {build.e_core_missing}")
    a("")

    a("## Per-file line / record / unique counts\n")
    a("| file | status | lines | records | new unique |")
    a("|---|---|---:|---:|---:|")
    tot_l = tot_r = tot_u = 0
    for fs in file_stats:
        a(f"| {fs.name} | {fs.status} | {fs.lines} | {fs.records} | {fs.new_unique} |")
        tot_l += fs.lines; tot_r += fs.records; tot_u += fs.new_unique
    a(f"| **total** | | {tot_l} | {tot_r} | {tot_u} |")
    a("")

    a("## Per-library record counts\n")
    a("| library_id | records | cbc_max harvested | cbc coverage |")
    a("|---|---:|---:|---:|")
    via = build.resolved_via
    for lib, n in sorted(build.library_counts.items(), key=lambda kv: -kv[1]):
        filled = build.library_cbc_filled.get(lib, 0)
        pct = 100.0 * filled / n if n else 0.0
        a(f"| {lib} | {n} | {filled} | {pct:.1f}% |")
    a(f"\nresolution source totals: run_meta={via.get('run_meta', 0)}, "
      f"name_pattern={via.get('name_pattern', 0)}")
    a("(5.8_5.1 / older-era case dirs were purged after caching -> boc_only; "
      "260624 and legacy_a case dirs survive.)\n")

    a("## Top-30 case pairs\n")
    a("| pair | records |")
    a("|---|---:|")
    for pair, n in sorted(build.pair_counts.items(), key=lambda kv: -kv[1])[:30]:
        a(f"| {pair} | {n} |")
    a(f"\ndistinct pairs: {len(build.pair_counts)}\n")

    a("## CBC recompute coverage\n")
    cov = 100.0 * build.cbc_max_filled / build.n_records if build.n_records else 0.0
    a(f"- cbc_kind=\"max\" (harvested): {build.cbc_max_filled} / {build.n_records} "
      f"= **{cov:.1f}%**")
    a(f"- cbc_kind=\"boc_only\" (residual): {build.n_records - build.cbc_max_filled}")
    a("")

    a("## (feed x e_core-bin) 2-D support histogram\n")
    feeds = sorted({f for f, _ in build.feed_ecore_hist})
    bins = sorted({b for _, b in build.feed_ecore_hist},
                  key=lambda s: (s == "nan", s))
    a("| feed \\ e_core | " + " | ".join(bins) + " |")
    a("|---|" + "|".join("---:" for _ in bins) + "|")
    for f in feeds:
        row = [str(build.feed_ecore_hist.get((f, b), 0)) for b in bins]
        a(f"| {f} | " + " | ".join(row) + " |")
    a("")

    a("## Harvest\n")
    a(f"- case dirs indexed: {n_case_dirs}")
    a("| status | count |")
    a("|---|---:|")
    for k, v in sorted(harvest_counts.items()):
        a(f"| {k} | {v} |")
    io_fail = sum(harvest_counts.get(k, 0) for k in ("inp_err", "sum_err", "no_cy"))
    a(f"\nharvest IO / parse failures: {io_fail}   "
      f"(missing MAS_SUM -> boc_only: {harvest_counts.get('no_sum', 0)})\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_extract_a(cfg: Any, *, limit: int | None = None, workers: int | None = None,
                  harvest_limit: int | None = None, progress: bool = True) -> dict[str, Any]:
    """Full Dataset A extraction: scan -> resolve -> harvest -> build -> write.

    ``limit`` caps the number of unique records (smoke runs); when set, the
    harvest is capped to ``harvest_limit`` (default ``20*limit``) dirs so the run
    stays fast.  Returns a stats dict for the caller to print.
    """
    t0 = time.time()
    deck_dir = (cfg.source_path.parent if getattr(cfg, "source_path", None)
                else Path.cwd())
    ex = cfg.extract
    workers = ex.workers if workers is None else workers
    store_dir = (deck_dir / ex.store_dir).resolve()
    reports_dir = (deck_dir / ex.reports_dir).resolve()

    workspaces = resolve_workspaces(cfg, deck_dir)
    if progress:
        print(f"[extract-a] workspaces: {[w.name for w in workspaces]}")

    # (a) scan + dedup
    raw, file_stats = scan_caches(workspaces, limit=limit, progress=progress)
    if progress:
        print(f"[extract-a] scanned -> {len(raw)} unique records")

    # (b) run_meta library index
    run_index = build_run_library_index(workspaces)
    if progress:
        print(f"[extract-a] run_meta library index: {len(run_index)} runs")

    # (d) harvest
    if limit is not None and harvest_limit is None:
        harvest_limit = 20 * limit
    case_dirs = index_case_dirs(workspaces, limit=harvest_limit)
    if progress:
        print(f"[extract-a] harvesting {len(case_dirs)} case dirs (workers={workers})")
    target_keys = {rr.dedup_key for rr in raw}
    harvested, harvest_counts = harvest(
        case_dirs, target_keys, workers=workers, progress=progress
    )
    if progress:
        print(f"[extract-a] harvested {len(harvested)} matched patterns")

    # (b/c/e) build records
    fuel_lib = FuelLibrary.from_parquet(store_dir / "fuel_types.parquet")
    records, maps_out, build = build_records(
        raw, run_index, fuel_lib, harvested, progress=progress
    )

    # (f) write store + report
    writer = StoreWriter(store_dir)
    rec_stats = writer.write_records(records, append=False)
    map_stats = writer.write_maps(maps_out, append=False) if maps_out else {"new": 0, "total": 0}
    wall = time.time() - t0
    report_path = reports_dir / "extract_report.md"
    write_report(report_path, file_stats, build, harvest_counts,
                 len(case_dirs), wall, store_dir)

    result = {
        "n_records": build.n_records,
        "library_counts": dict(build.library_counts),
        "unresolved": build.unresolved,
        "cbc_max_filled": build.cbc_max_filled,
        "cbc_coverage_pct": (100.0 * build.cbc_max_filled / build.n_records
                             if build.n_records else 0.0),
        "footprints": build.footprints,
        "converged": build.converged,
        "nonconverged": build.nonconverged,
        "harvest_counts": harvest_counts,
        "n_case_dirs": len(case_dirs),
        "records_written": rec_stats["total"],
        "maps_written": map_stats["total"],
        "wall_s": wall,
        "store_dir": str(store_dir),
        "report_path": str(report_path),
    }
    return result
