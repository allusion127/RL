"""Dataset B extraction pipeline (plan 4.2, milestone M2-B).

Dataset B is the 3_GA_Surrogate corpus: the GA-search event logs plus the
hand-/tool-packaged warm-seed manifests.  Unlike Dataset A (one self-describing
cache), the FOM labels and the loading patterns live in *separate* artefacts and
must be re-joined, and most of the upstream is OneDrive-dehydrated.

Stages (each a function; orchestrated by :func:`run_extract_b`):

  1. **event-log scan** — every ``runs_flow/*/stages/ga_generations_*.jsonl``.
     Each JSONL line is a GA generation whose ``batch`` list holds per-candidate
     entries: ``digest`` (vendor 16-hex ``Pattern.digest``), ``fom`` dict,
     ``eq_ok`` / ``feasible`` flags, ``parent_digest``, ``selected`` (generator),
     or an ``error`` string.  Deduplicated by ``(case, digest)`` (a converged
     ``fom`` entry beats an ``error`` re-evaluation of the same pattern).
  2. **pattern recovery** — a ``digest -> Pattern`` index built by parsing every
     readable ``%LPD_SHF`` deck under ``runs_flow`` (``loading_shf.txt`` +
     ``MAS_INP*.inp``) and recomputing the vendor digest.  Event-log labels are
     *only trainable with a pattern*; unrecovered labels are counted and skipped
     (they lived in purged GA worker dirs and will re-appear when the dehydrated
     manifests hydrate).
  3. **manifest scan** — ``manifest.csv`` + ``cores/<case>/<id>/loading_shf.txt``
     of each package root; every row is self-contained (pattern + FOM).  All
     roots are dehydrated today: attempted, per-file caught, counted, skipped.
  4. **record build** — schema mapping + derived columns (feed, n_batches /
     depth2 via :class:`GeneralOrbitGenome`, e_core / e_split via the ga80
     :class:`FuelLibrary`), parent lineage resolved against ingested rows.
  5. **write** — :class:`StoreWriter` (dedup vs Dataset A is automatic via
     ``record_id``; ga80 keeps B ids distinct) + a Dataset B section appended to
     ``data/reports/extract_report.md``.

Windows/Korean paths: pathlib throughout, utf-8-sig with cp949 fallback.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..search.genome import GeneralOrbitGenome, GenomeError
from ..vendor.masterrl.domain import Pattern
from .extract_a import _e_core_split, _ecore_bin, _read_text_flex, _type_resolver
from .fuel_types import FuelLibrary
from .geometry import to_canonical_from_shf
from .schema import CanonicalRecord, compute_record_id
from .store import StoreWriter

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: Fixed fuel library of every Dataset B record (the ga80 letter library).
GA80_LIBRARY = "ga80"

#: Symmetry class of Dataset B records (feed-general orbit representation).
GA_SYM_CLASS = "free69"

#: Deck-knob signature of the ga80 MASTER harness (record_id component; distinct
#: from Dataset A's ``mocha_default`` — though library_id already separates them).
GA_DECK_KNOBS = "ga_native"

#: restart_provenance tag of every Dataset B record.
GA_RESTART_PROVENANCE = "ga_native"

_CASE_RE = re.compile(r"^(?P<pair>[^/]+?)(?:/feed-(?P<feed>\d+))?$")
_EVENT_LOG_GLOB = "*/stages/ga_generations_*.jsonl"


# --------------------------------------------------------------------------- #
# case-string parsing
# --------------------------------------------------------------------------- #
def parse_case(case: str) -> tuple[str, int | None]:
    """``"K1_K2/feed-121"`` -> ``("K1_K2", 121)``; feed is None if absent."""
    m = _CASE_RE.match(case.strip())
    if not m:
        return case.strip(), None
    feed = m.group("feed")
    return m.group("pair"), (int(feed) if feed is not None else None)


# --------------------------------------------------------------------------- #
# FOM -> schema metric mapping
# --------------------------------------------------------------------------- #
def _f(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _ao_abs(d: dict) -> float | None:
    lo, hi = _f(d, "AO_min"), _f(d, "AO_max")
    if lo is None and hi is None:
        return None
    return max(abs(lo or 0.0), abs(hi or 0.0))


def map_fom(fom: dict) -> dict[str, float | None]:
    """Vendor FOM dict (event log or manifest row) -> schema target fields.

    3_GA's ``CBC_max`` is the EDIT2 max (plan 1-5), so it feeds ``cbc_max`` with
    ``cbc_kind="max"``; ``cbc_boc`` is not carried by either source (left None).
    """
    return {
        "f_r": _f(fom, "F_r"),
        "f_q": _f(fom, "F_q"),
        "cbc_max": _f(fom, "CBC_max"),
        "cyclen": _f(fom, "cyclen"),
        "ao_abs": _ao_abs(fom),
        "max_assembly_burnup": _f(fom, "max_assembly_burnup"),
        "max_pin_burnup": _f(fom, "max_pin_burnup"),
    }


# --------------------------------------------------------------------------- #
# stage 1: event-log scan + dedup
# --------------------------------------------------------------------------- #
@dataclass
class EventLabel:
    """One deduplicated GA candidate label (FOM only; pattern joined later)."""

    case_pair: str
    feed: int | None
    digest: str
    campaign: str                 # runs_flow timestamp
    generator: str | None         # the GA ``selected`` bucket
    parent_digest: str | None
    fom: dict[str, Any]
    eq_ok: bool
    feasible: bool
    error: str | None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass
class RunStat:
    """Raw (pre-dedup) per-``(run, case)`` counts for the report."""

    campaign: str
    case: str
    entries: int = 0
    feasible: int = 0
    errors: int = 0
    eq_ok: int = 0


def scan_event_logs(runs_flow: Path, *, progress: bool = True
                    ) -> tuple[dict[tuple[str, str], EventLabel], list[RunStat], int]:
    """Scan + dedup every ``ga_generations_*.jsonl`` under ``runs_flow``.

    Returns ``(labels_by_case_digest, per_run_stats, total_entries)``.  Dedup is
    by ``(case_pair, digest)``; a converged (``fom``-bearing) entry replaces an
    ``error`` re-evaluation of the same pattern.
    """
    labels: dict[tuple[str, str], EventLabel] = {}
    run_stats: list[RunStat] = []
    total_entries = 0

    logs = sorted(runs_flow.glob(_EVENT_LOG_GLOB)) if runs_flow.is_dir() else []
    for log in logs:
        campaign = log.parent.parent.name          # runs_flow/<ts>/stages/<log>
        try:
            text = _read_text_flex(log)
        except OSError:
            continue
        per_case: dict[str, RunStat] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            case = str(obj.get("case", ""))
            pair, feed = parse_case(case)
            stat = per_case.setdefault(case, RunStat(campaign, case))
            for e in obj.get("batch", []) or []:
                digest = e.get("digest")
                if not digest:
                    continue
                total_entries += 1
                stat.entries += 1
                fom = e.get("fom") if isinstance(e.get("fom"), dict) else None
                error = e.get("error")
                feasible = bool(e.get("feasible", False))
                eq_ok = bool(e.get("eq_ok", False))
                if feasible:
                    stat.feasible += 1
                if eq_ok:
                    stat.eq_ok += 1
                if error is not None and fom is None:
                    stat.errors += 1

                label = EventLabel(
                    case_pair=pair,
                    feed=feed,
                    digest=str(digest),
                    campaign=campaign,
                    generator=(str(e["selected"]) if e.get("selected") else None),
                    parent_digest=(str(e["parent_digest"]) if e.get("parent_digest") else None),
                    fom=fom or {},
                    eq_ok=eq_ok,
                    feasible=feasible,
                    error=(str(error) if (error is not None and fom is None) else None),
                )
                key = (pair, label.digest)
                prior = labels.get(key)
                # A converged label beats an error re-evaluation of the pattern.
                if prior is None or (prior.is_error and not label.is_error):
                    labels[key] = label
        run_stats.extend(per_case.values())
        if progress:
            tot = sum(s.entries for s in per_case.values())
            print(f"  event-log {campaign}/{log.name[:36]:36s} entries={tot}")

    return labels, run_stats, total_entries


# --------------------------------------------------------------------------- #
# stage 2: pattern recovery (digest -> Pattern)
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryStat:
    files_scanned: int = 0
    read_errors: int = 0
    parse_errors: int = 0
    unique_digests: int = 0
    dehydrated_roots: list[str] = field(default_factory=list)


def _parse_deck(path: Path) -> Pattern | None:
    """Parse one ``%LPD_SHF`` deck to a :class:`Pattern`; None on read/parse fail.

    Read failures propagate as :class:`OSError` (so callers can distinguish a
    dehydrated placeholder from a malformed deck).
    """
    text = _read_text_flex(path)                    # OSError -> caller counts it
    try:
        return to_canonical_from_shf(text)
    except (ValueError, KeyError, AssertionError):
        return None


def build_pattern_index(runs_flow: Path, manifest_roots: list[Path], *,
                        progress: bool = True) -> tuple[dict[str, Pattern], RecoveryStat]:
    """Build ``digest -> Pattern`` from every readable ``%LPD_SHF`` deck.

    Scans ``runs_flow`` decks (``loading_shf.txt`` + ``MAS_INP*.inp``) and also
    attempts each manifest root's ``cores/**/loading_shf.txt`` (dehydrated today;
    each unreadable root is recorded and skipped).
    """
    index: dict[str, Pattern] = {}
    stat = RecoveryStat()

    def _ingest(path: Path) -> None:
        stat.files_scanned += 1
        try:
            pat = _parse_deck(path)
        except OSError:
            stat.read_errors += 1
            return
        if pat is None:
            stat.parse_errors += 1
            return
        index[pat.digest] = pat

    if runs_flow.is_dir():
        for name in ("loading_shf.txt", "MAS_INP*.inp"):
            for path in runs_flow.rglob(name):
                _ingest(path)

    for root in manifest_roots:
        cores = root / "cores"
        if not cores.is_dir():
            continue
        readable = False
        for path in cores.rglob("loading_shf.txt"):
            before = stat.read_errors
            _ingest(path)
            if stat.read_errors == before:
                readable = True
        if not readable and cores.is_dir():
            stat.dehydrated_roots.append(root.name)

    stat.unique_digests = len(index)
    if progress:
        print(f"  pattern index: {stat.unique_digests} unique digests "
              f"(files={stat.files_scanned} read_err={stat.read_errors} "
              f"parse_err={stat.parse_errors})")
    return index, stat


# --------------------------------------------------------------------------- #
# stage 3: manifest scan (dehydrated-tolerant)
# --------------------------------------------------------------------------- #
_MANIFEST_REQUIRED = ("id", "pair", "feed", "cell", "F_r", "CBC_max", "F_q", "cyclen", "ncyc")


@dataclass
class ManifestLabel:
    """One manifest seed row: pattern + FOM (self-contained, no digest join)."""

    case_pair: str
    feed: int
    campaign: str                 # manifest root name
    pattern: Pattern
    fom: dict[str, Any]
    eq_ok: bool
    n_cycles: float | None


@dataclass
class ManifestStat:
    root: str
    status: str                   # OK | DEHYDRATED | NO_MANIFEST | ERROR
    rows: int = 0
    joined: int = 0
    row_errors: int = 0
    detail: str = ""


def _folder_of(pair: str, feed: int) -> str:
    """FEASIBLE_PACKAGE core folder name: ``pair`` at feed 121 else ``pair_f<feed>``."""
    return pair if feed == 121 else f"{pair}_f{feed}"


def scan_manifest(root: Path) -> tuple[list[ManifestLabel], ManifestStat]:
    """Read one package root's ``manifest.csv`` + per-seed ``loading_shf.txt``.

    Fully tolerant of OneDrive dehydration: a manifest that exists-but-fails-to-
    read is reported ``DEHYDRATED`` with zero rows; a readable manifest whose
    per-seed patterns are dehydrated counts those as ``row_errors`` and skips.
    """
    stat = ManifestStat(root=root.name, status="NO_MANIFEST")
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        return [], stat
    try:
        text = _read_text_flex(manifest)
    except OSError as exc:
        stat.status = "DEHYDRATED"
        stat.detail = f"manifest unreadable: {exc}"
        return [], stat

    try:
        reader = csv.DictReader(text.splitlines())
        rows = list(reader)
        cols = set(reader.fieldnames or ())
    except (csv.Error, ValueError) as exc:
        stat.status = "ERROR"
        stat.detail = f"csv parse: {exc}"
        return [], stat

    missing = set(_MANIFEST_REQUIRED) - cols
    if missing:
        stat.status = "ERROR"
        stat.detail = f"missing columns {sorted(missing)}"
        return [], stat

    labels: list[ManifestLabel] = []
    stat.status = "OK"
    for row in rows:
        stat.rows += 1
        try:
            pair = str(row["pair"]).strip()
            feed = int(row["feed"])
            seed_id = str(row["id"]).strip()
        except (KeyError, TypeError, ValueError):
            stat.row_errors += 1
            continue
        shf = root / "cores" / _folder_of(pair, feed) / seed_id / "loading_shf.txt"
        if not shf.is_file():
            stat.row_errors += 1
            continue
        try:
            pat = _parse_deck(shf)
        except OSError:
            stat.row_errors += 1                    # dehydrated seed
            continue
        if pat is None:
            stat.row_errors += 1
            continue
        fom = {
            "F_r": row.get("F_r"), "CBC_max": row.get("CBC_max"),
            "F_q": row.get("F_q"), "cyclen": row.get("cyclen"),
            "AO_min": row.get("AO_min"), "AO_max": row.get("AO_max"),
            "max_assembly_burnup": row.get("max_assembly_burnup"),
            "max_pin_burnup": row.get("max_pin_burnup"),
        }
        eq_raw = str(row.get("eq_ok", "") or "").strip().lower()
        eq_ok = eq_raw in ("", "true", "1")        # default True (vendor semantics)
        ncyc = row.get("ncyc")
        try:
            n_cycles = float(ncyc) if ncyc not in (None, "") else None
        except (TypeError, ValueError):
            n_cycles = None
        labels.append(ManifestLabel(
            case_pair=pair, feed=feed, campaign=root.name, pattern=pat,
            fom=fom, eq_ok=eq_ok, n_cycles=n_cycles,
        ))
        stat.joined += 1
    return labels, stat


# --------------------------------------------------------------------------- #
# stage 4: build canonical records
# --------------------------------------------------------------------------- #
@dataclass
class BuildStats:
    n_records: int = 0
    event_recovered: int = 0
    event_unrecovered: int = 0
    manifest_records: int = 0
    converged: int = 0
    nonconverged: int = 0
    error_rows: int = 0
    feed_mismatch: int = 0
    pair_mismatch: int = 0
    genome_failures: int = 0
    e_core_missing: int = 0
    parent_resolved: int = 0
    parent_unresolved: int = 0
    pair_counts: dict[str, int] = field(default_factory=dict)
    feed_ecore_hist: dict[tuple[int, str], int] = field(default_factory=dict)


def _derive(pattern: Pattern, fuel_lib: FuelLibrary, resolve_type,
            stats: BuildStats) -> tuple[int, int, float | None, float | None]:
    """(n_batches, depth2_edges, e_core, e_split) for a pattern's ga80 core."""
    try:
        genome = GeneralOrbitGenome.from_pattern(pattern)
        depth2 = int(genome.depth2_edge_count)
        n_batches = 2 if depth2 == 0 else 3
    except (GenomeError, ValueError):
        stats.genome_failures += 1
        depth2, n_batches = 0, 2
    e_core, e_split = _e_core_split(
        fuel_lib, GA80_LIBRARY, pattern.batch_feed(), resolve_type
    )
    if e_core is None:
        stats.e_core_missing += 1
    return n_batches, depth2, e_core, e_split


def _make_record(*, pattern: Pattern, case_pair: str, campaign: str,
                 generator: str | None, parent_record_id: str | None,
                 metrics: dict[str, float | None], converged: bool, valid: bool,
                 failure: str, n_cycles: float | None, fuel_lib: FuelLibrary,
                 resolve_type, stats: BuildStats) -> tuple[str, CanonicalRecord]:
    canonical = pattern.canonical()
    feed = pattern.feed
    n_batches, depth2, e_core, e_split = _derive(pattern, fuel_lib, resolve_type, stats)
    record_id = compute_record_id(canonical, GA80_LIBRARY, case_pair, GA_DECK_KNOBS)

    have_cbc = metrics.get("cbc_max") is not None
    rec = CanonicalRecord(
        record_id=record_id,
        dataset="B",
        campaign=campaign,
        stratum=None,
        generator=generator,
        parent_record_id=parent_record_id,
        case_pair=case_pair,
        feed=feed,
        n_batches=n_batches,
        depth2_edges=depth2,
        e_core=e_core,
        e_split=e_split,
        library_id=GA80_LIBRARY,
        sym_class=GA_SYM_CLASS,
        pattern=canonical,
        f_r=metrics.get("f_r"),
        f_q=metrics.get("f_q"),
        cbc_max=metrics.get("cbc_max"),
        cbc_boc=None,
        cbc_kind="max" if have_cbc else "boc_only",
        cyclen=metrics.get("cyclen"),
        ao_abs=metrics.get("ao_abs"),
        cycle_burnup=None,
        discharge_burnup=None,
        max_assembly_burnup=metrics.get("max_assembly_burnup"),
        max_pin_burnup=metrics.get("max_pin_burnup"),
        eoc_ppm=None,
        delta_efpd=None,
        n_cycles=n_cycles,
        converged=converged,
        converged_at_cap=False,
        tolerance_margin=None,
        restart_provenance=GA_RESTART_PROVENANCE,
        valid=valid,
        failure=failure,
        maps_key=None,
    )

    stats.pair_counts[case_pair] = stats.pair_counts.get(case_pair, 0) + 1
    binkey = (feed, _ecore_bin(e_core))
    stats.feed_ecore_hist[binkey] = stats.feed_ecore_hist.get(binkey, 0) + 1
    if converged:
        stats.converged += 1
    else:
        stats.nonconverged += 1
    return record_id, rec


def build_records(labels: dict[tuple[str, str], EventLabel],
                  pattern_index: dict[str, Pattern],
                  manifest_labels: list[ManifestLabel],
                  fuel_lib: FuelLibrary, *, progress: bool = True
                  ) -> tuple[list[CanonicalRecord], BuildStats]:
    """Join labels to patterns and assemble :class:`CanonicalRecord`s.

    Event-log labels without a recovered pattern are dropped (counted as
    ``event_unrecovered``); manifest labels carry their own pattern.  Parent
    lineage is resolved in a second pass against the ingested rows only.
    """
    resolve_type = _type_resolver(fuel_lib)
    stats = BuildStats()
    records: list[CanonicalRecord] = []

    #: digest -> record_id for ingested event rows (for parent lineage).
    digest_to_rid: dict[str, str] = {}
    #: index of event records that carry a parent_digest, for the 2nd pass.
    pending_parents: list[tuple[int, str]] = []

    # -- event-log rows --------------------------------------------------------
    for (pair, digest), label in labels.items():
        pattern = pattern_index.get(digest)
        if pattern is None:
            stats.event_unrecovered += 1
            continue
        stats.event_recovered += 1

        # Cross-check the recovered pattern against the campaign's declared case.
        if label.feed is not None and pattern.feed != label.feed:
            stats.feed_mismatch += 1
        derived_pair = "_".join(sorted(pattern.batch_feed()))
        if derived_pair != pair:
            stats.pair_mismatch += 1

        if label.is_error:
            metrics = {k: None for k in
                       ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                        "max_assembly_burnup", "max_pin_burnup")}
            converged, valid, failure = False, False, label.error or ""
            stats.error_rows += 1
        else:
            metrics = map_fom(label.fom)
            converged, valid, failure = label.eq_ok, True, ""

        rid, rec = _make_record(
            pattern=pattern, case_pair=pair, campaign=label.campaign,
            generator=label.generator, parent_record_id=None, metrics=metrics,
            converged=converged, valid=valid, failure=failure, n_cycles=None,
            fuel_lib=fuel_lib, resolve_type=resolve_type, stats=stats,
        )
        digest_to_rid[digest] = rid
        if label.parent_digest:
            pending_parents.append((len(records), label.parent_digest))
        records.append(rec)

    # -- parent lineage (resolve only against ingested rows) -------------------
    for idx, parent_digest in pending_parents:
        parent_rid = digest_to_rid.get(parent_digest)
        if parent_rid is not None:
            records[idx] = replace(records[idx], parent_record_id=parent_rid)
            stats.parent_resolved += 1
        else:
            stats.parent_unresolved += 1

    # -- manifest rows ---------------------------------------------------------
    for ml in manifest_labels:
        metrics = map_fom(ml.fom)
        _rid, rec = _make_record(
            pattern=ml.pattern, case_pair=ml.case_pair, campaign=ml.campaign,
            generator=None, parent_record_id=None, metrics=metrics,
            converged=ml.eq_ok, valid=True, failure="", n_cycles=ml.n_cycles,
            fuel_lib=fuel_lib, resolve_type=resolve_type, stats=stats,
        )
        stats.manifest_records += 1
        records.append(rec)

    stats.n_records = len(records)
    if progress:
        print(f"  built {stats.n_records} records "
              f"(event {stats.event_recovered}, manifest {stats.manifest_records}, "
              f"unrecovered {stats.event_unrecovered})")
    return records, stats


# --------------------------------------------------------------------------- #
# stage 5: report
# --------------------------------------------------------------------------- #
#: Marker delimiting the Dataset B section so a B re-run replaces (not appends).
_B_MARKER = "<!-- lpopt:dataset-b -->"


def render_report(run_stats: list[RunStat], total_entries: int, n_unique: int,
                  recovery: RecoveryStat, manifest_stats: list[ManifestStat],
                  build: BuildStats, wall_s: float, store_dir: Path) -> str:
    """Render the Dataset B report section (markdown, no leading marker)."""
    lines: list[str] = []
    a = lines.append
    a("# Dataset B extraction report\n")
    a(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- wall time: {wall_s:.1f} s")
    a(f"- store: `{store_dir}`")
    a(f"- event-log entries (raw): {total_entries}   unique (case,digest): {n_unique}")
    a(f"- records written: **{build.n_records}**  "
      f"(event {build.event_recovered} + manifest {build.manifest_records})")
    a(f"- converged: {build.converged}   non-converged: {build.nonconverged}   "
      f"error rows: {build.error_rows}")
    rate = (100.0 * build.event_recovered / (build.event_recovered + build.event_unrecovered)
            if (build.event_recovered + build.event_unrecovered) else 0.0)
    a(f"- pattern recovery: {build.event_recovered} / "
      f"{build.event_recovered + build.event_unrecovered} = **{rate:.1f}%** "
      f"(unrecovered dropped: {build.event_unrecovered})")
    a(f"- parent lineage: {build.parent_resolved} resolved, "
      f"{build.parent_unresolved} unresolved (parent not ingested)")
    a(f"- feed / pair mismatches (recovered vs case string): "
      f"{build.feed_mismatch} / {build.pair_mismatch}")
    a(f"- genome parse failures: {build.genome_failures}   "
      f"e_core missing: {build.e_core_missing}")
    a("")

    a("## Per-run event-log counts (raw, pre-dedup)\n")
    a("| campaign (runs_flow ts) | case | entries | feasible | errors | eq_ok |")
    a("|---|---|---:|---:|---:|---:|")
    for s in sorted(run_stats, key=lambda r: (r.campaign, r.case)):
        a(f"| {s.campaign} | {s.case} | {s.entries} | {s.feasible} | "
          f"{s.errors} | {s.eq_ok} |")
    a("")
    # Audit ground truth call-out (plan M2 / sec 1-10).
    audit = [s for s in run_stats
             if s.campaign == "20260713_061541" and s.case.startswith("K1_K2")]
    if audit:
        s = audit[0]
        ok = (s.entries == 600 and s.feasible == 70 and s.errors == 17)
        a(f"**Audit run 20260713_061541 / K1_K2**: {s.entries} labels / "
          f"{s.feasible} feasible / {s.errors} errors "
          f"(ground truth 600 / 70 / 17 -> {'MATCH' if ok else 'MISMATCH'})\n")

    a("## Per-pair record counts\n")
    a("| case_pair | records |")
    a("|---|---:|")
    for pair, n in sorted(build.pair_counts.items(), key=lambda kv: -kv[1]):
        a(f"| {pair} | {n} |")
    a("")

    a("## Pattern recovery\n")
    a(f"- deck files scanned: {recovery.files_scanned}")
    a(f"- read errors (dehydrated placeholders): {recovery.read_errors}")
    a(f"- parse errors: {recovery.parse_errors}")
    a(f"- unique recovered digests: {recovery.unique_digests}")
    if recovery.dehydrated_roots:
        a(f"- dehydrated manifest-core roots skipped: {recovery.dehydrated_roots}")
    a("")

    a("## Manifests (dehydrated-tolerant)\n")
    a("| root | status | rows | joined | row errors | detail |")
    a("|---|---|---:|---:|---:|---|")
    for m in manifest_stats:
        a(f"| {m.root} | {m.status} | {m.rows} | {m.joined} | "
          f"{m.row_errors} | {m.detail} |")
    a("")

    a("## (feed x e_core-bin) 2-D support histogram\n")
    feeds = sorted({f for f, _ in build.feed_ecore_hist})
    bins = sorted({b for _, b in build.feed_ecore_hist}, key=lambda s: (s == "nan", s))
    if feeds and bins:
        a("| feed \\ e_core | " + " | ".join(bins) + " |")
        a("|---|" + "|".join("---:" for _ in bins) + "|")
        for f in feeds:
            row = [str(build.feed_ecore_hist.get((f, b), 0)) for b in bins]
            a(f"| {f} | " + " | ".join(row) + " |")
    else:
        a("(no records)")
    a("")
    return "\n".join(lines)


def write_report(path: Path, section: str) -> None:
    """Append/replace the Dataset B section of ``extract_report.md`` in place.

    The Dataset A section (written by :mod:`extract_a`) is preserved: everything
    before :data:`_B_MARKER` is kept and the B section after it is regenerated,
    so repeated B runs never duplicate or clobber the A report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        prefix = existing.split(_B_MARKER)[0].rstrip() + "\n\n"
    path.write_text(prefix + _B_MARKER + "\n\n" + section, encoding="utf-8")


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_extract_b(cfg: Any, *, limit: int | None = None,
                  progress: bool = True) -> dict[str, Any]:
    """Full Dataset B extraction: event logs + manifests -> store + report.

    ``limit`` caps the number of event-log labels considered (smoke runs).
    Returns a stats dict for the caller to print.
    """
    t0 = time.time()
    deck_dir = (cfg.source_path.parent if getattr(cfg, "source_path", None)
                else Path.cwd())
    ex = cfg.extract
    store_dir = (deck_dir / ex.store_dir).resolve()
    reports_dir = (deck_dir / ex.reports_dir).resolve()

    ga_root = Path(ex.ga_root)
    ga_root = ga_root if ga_root.is_absolute() else (deck_dir / ga_root)
    ga_root = ga_root.resolve()
    runs_flow = ga_root / ex.ga_runs_flow
    manifest_roots = [ga_root / r for r in ex.ga_manifest_roots]

    if progress:
        print(f"[extract-b] ga_root: {ga_root}")
        print(f"[extract-b] runs_flow: {runs_flow}  (exists={runs_flow.is_dir()})")

    # (1) event logs
    labels, run_stats, total_entries = scan_event_logs(runs_flow, progress=progress)
    if limit is not None:
        labels = dict(list(labels.items())[:limit])
    n_unique = len(labels)
    if progress:
        print(f"[extract-b] event-log unique (case,digest): {n_unique}")

    # (2) pattern recovery
    pattern_index, recovery = build_pattern_index(
        runs_flow, manifest_roots, progress=progress
    )

    # (3) manifests
    manifest_labels: list[ManifestLabel] = []
    manifest_stats: list[ManifestStat] = []
    for root in manifest_roots:
        mls, mstat = scan_manifest(root)
        manifest_labels.extend(mls)
        manifest_stats.append(mstat)
        if progress:
            print(f"  manifest {mstat.root:20s} {mstat.status:12s} "
                  f"rows={mstat.rows} joined={mstat.joined}")

    # (4) build records
    fuel_lib = FuelLibrary.from_parquet(store_dir / "fuel_types.parquet")
    records, build = build_records(
        labels, pattern_index, manifest_labels, fuel_lib, progress=progress
    )

    # (5) write store + report (append to store; dedup vs Dataset A by record_id)
    writer = StoreWriter(store_dir)
    rec_stats = writer.write_records(records, append=True)
    wall = time.time() - t0
    section = render_report(
        run_stats, total_entries, n_unique, recovery, manifest_stats,
        build, wall, store_dir,
    )
    report_path = reports_dir / "extract_report.md"
    write_report(report_path, section)

    audit = next((s for s in run_stats if s.campaign == "20260713_061541"
                  and s.case.startswith("K1_K2")), None)
    result = {
        "n_records": build.n_records,
        "event_recovered": build.event_recovered,
        "event_unrecovered": build.event_unrecovered,
        "manifest_records": build.manifest_records,
        "n_unique_labels": n_unique,
        "total_entries": total_entries,
        "recovery_pct": (100.0 * build.event_recovered
                         / (build.event_recovered + build.event_unrecovered)
                         if (build.event_recovered + build.event_unrecovered) else 0.0),
        "converged": build.converged,
        "nonconverged": build.nonconverged,
        "error_rows": build.error_rows,
        "parent_resolved": build.parent_resolved,
        "parent_unresolved": build.parent_unresolved,
        "pair_counts": dict(build.pair_counts),
        "dehydrated_roots": recovery.dehydrated_roots,
        "manifest_stats": [(m.root, m.status, m.rows, m.joined) for m in manifest_stats],
        "audit_K1_K2": (
            {"entries": audit.entries, "feasible": audit.feasible, "errors": audit.errors}
            if audit else None
        ),
        "records_written_total": rec_stats["total"],
        "records_new": rec_stats["new"],
        "wall_s": wall,
        "store_dir": str(store_dir),
        "report_path": str(report_path),
    }
    return result
