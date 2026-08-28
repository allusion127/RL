"""Command-line interface for lpopt (``python -m lpopt <cmd>`` / ``lpopt <cmd>``).

Milestone M0 implements two commands fully:

* ``vendor-check`` — re-hash every vendored master_rl file against
  ``VENDOR_MANIFEST.json`` (integrity) and against its recorded source path
  (drift report only).  Integrity mismatch exits 1; drift alone exits 0.
* ``check`` — preflight every configured asset: existence AND an open-and-read
  of the first 64 KiB (to catch OneDrive *dehydrated placeholders*, which pass an
  existence test but fail to read).  Template decks are additionally checked for
  ``%LPD_SHF``, a ``%LPD_B&C`` card with >= 80 batch rows, and the *absence* of
  any feed-count token.  Exits 1 on any failure.

All other sub-commands (``extract``, ``produce``, ``train``, ``eval``,
``optimize``, ``report``, ``remote``) are stubs that print the milestone that
will implement them and exit 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, LpoptConfig, load_config

_READ_PROBE_BYTES = 64 * 1024
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "masterrl"
_MANIFEST = _VENDOR_DIR / "VENDOR_MANIFEST.json"

# Sub-command -> milestone that will implement it.  (All M4 targets are now
# wired; nothing remains stubbed.)
_STUBS: dict[str, str] = {}

# Statuses that count as a preflight failure (exit 1).
_FAIL_STATUSES = {"FAIL", "MISSING", "DEHYDRATED"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_text_flex(path: Path) -> str:
    """Read text with utf-8-sig, falling back to cp949 (Windows/Korean decks)."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


@dataclass
class ProbeResult:
    status: str  # PASS | MISSING | DEHYDRATED | FAIL | SKIP
    detail: str


def _probe_read(path: Path) -> ProbeResult:
    """Existence + open-and-read first 64 KiB (dehydrated-placeholder guard)."""
    if not path.exists():
        return ProbeResult("MISSING", "does not exist")
    if path.is_dir():
        return ProbeResult("PASS", "directory exists")
    try:
        with open(path, "rb") as handle:
            data = handle.read(_READ_PROBE_BYTES)
    except OSError as exc:
        return ProbeResult(
            "DEHYDRATED",
            f"exists but unreadable (cloud placeholder? provider must hydrate): {exc}",
        )
    if not data:
        return ProbeResult("FAIL", "zero bytes read (empty or placeholder)")
    return ProbeResult("PASS", f"read {len(data)} bytes")


def _resolve(path_str: str, base: Path) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (base / candidate)


def _count_bc_rows(text: str) -> int:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("%LPD_B&C"):
            start = i
            break
    if start is None:
        return 0
    count = 0
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            break
        count += 1
    return count


def _has_feed_token(text: str) -> bool:
    """True if the deck encodes a feed count directly (should never happen)."""
    return re.search(r"%\s*(LPD_)?FEED\b", text, re.IGNORECASE) is not None


# --------------------------------------------------------------------------- #
# vendor-check
# --------------------------------------------------------------------------- #
def cmd_vendor_check(_args: argparse.Namespace) -> int:
    if not _MANIFEST.exists():
        print(f"[ERROR] vendor manifest not found: {_MANIFEST}")
        return 1
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    files: dict[str, dict] = manifest["files"]

    print(f"vendor-check  ({_VENDOR_DIR})")
    print(f"pinned_snapshot: {manifest.get('pinned_snapshot', '?')}")
    print(f"copied_at:       {manifest.get('copied_at', '?')}")
    print()
    header = f"{'FILE':16s} {'INTEGRITY':10s} {'DRIFT vs source':28s} DETAIL"
    print(header)
    print("-" * len(header))

    integrity_failed = False
    for name in sorted(files):
        meta = files[name]
        vpath = _VENDOR_DIR / name
        detail = ""

        # Integrity: vendored file on disk vs manifest hash.
        if not vpath.exists():
            integrity = "MISSING"
            vhash = None
            integrity_failed = True
        else:
            vhash = _sha256(vpath)
            if vhash == meta["sha256"]:
                integrity = "OK"
            else:
                integrity = "MISMATCH"
                integrity_failed = True
                detail = "on-disk hash != manifest"

        # Drift: recorded source path vs what we vendored.
        note = str(meta.get("note", ""))
        is_shim = note.startswith("shim")
        is_patched = note.startswith("patched")
        src = Path(meta["source_path"])
        if is_patched:
            # A deliberate local delta from the pinned snapshot (documented in
            # the manifest note); source drift is not meaningful here.
            drift = "n/a (local patch)"
        elif is_shim:
            drift = "n/a (shim subset)"
        elif not src.exists():
            drift = "source missing"
        else:
            try:
                shash = _sha256(src)
            except OSError:
                drift = "source unreadable (dehydrated?)"
            else:
                if vhash is not None and shash == vhash:
                    drift = "in-sync"
                else:
                    drift = "DRIFT (source changed)"

        print(f"{name:16s} {integrity:10s} {drift:28s} {detail}")

    print()
    if integrity_failed:
        print("RESULT: FAIL — vendored file(s) do not match manifest (integrity).")
        return 1
    print("RESULT: OK — all vendored files match the manifest. (drift is report-only)")
    return 0


# --------------------------------------------------------------------------- #
# check (preflight)
# --------------------------------------------------------------------------- #
@dataclass
class CheckItem:
    group: str
    item: str
    status: str
    detail: str


def _check_template_deck(path: Path) -> ProbeResult:
    probe = _probe_read(path)
    if probe.status != "PASS":
        return probe
    try:
        text = _read_text_flex(path)
    except OSError as exc:
        return ProbeResult("DEHYDRATED", f"unreadable: {exc}")
    problems: list[str] = []
    notes: list[str] = []
    if "%LPD_SHF" in text:
        notes.append("%LPD_SHF ok")
    else:
        problems.append("missing %LPD_SHF")
    n_rows = _count_bc_rows(text)
    if n_rows >= 80:
        notes.append(f"{n_rows} B&C rows")
    else:
        problems.append(f"%LPD_B&C has {n_rows} rows (< 80)")
    if _has_feed_token(text):
        problems.append("feed-count token present")
    else:
        notes.append("no feed token")
    if problems:
        return ProbeResult("FAIL", "; ".join(problems))
    return ProbeResult("PASS", "; ".join(notes))


def _collect_checks(cfg: LpoptConfig, deck_dir: Path) -> list[CheckItem]:
    items: list[CheckItem] = []

    # --- MASTER executable ---------------------------------------------------
    if cfg.master.executable:
        exe = _resolve(cfg.master.executable, deck_dir)
        probe = _probe_read(exe)
        items.append(CheckItem("master.executable", str(exe), probe.status, probe.detail))
    else:
        items.append(
            CheckItem("master.executable", "(unset)", "SKIP", "no [master].executable")
        )

    # --- verify.package_root -------------------------------------------------
    if cfg.verify.package_root:
        root = _resolve(cfg.verify.package_root, deck_dir)
        if not root.exists():
            items.append(
                CheckItem("verify.package_root", str(root), "MISSING", "root does not exist")
            )
        else:
            items.append(
                CheckItem("verify.package_root", str(root), "PASS", "root exists")
            )
            # lib/MAS_XSL, lib/MAS_HFF
            for lib_name in ("lib/MAS_XSL", "lib/MAS_HFF"):
                p = root / lib_name
                probe = _probe_read(p)
                items.append(CheckItem("verify.lib", str(p), probe.status, probe.detail))
            # bases/*/MAS_RST.*
            rst_files = sorted(root.glob("bases/*/MAS_RST.*"))
            if not rst_files:
                items.append(
                    CheckItem("verify.bases", str(root / "bases"), "FAIL", "no bases/*/MAS_RST.* found")
                )
            else:
                for p in rst_files:
                    probe = _probe_read(p)
                    items.append(CheckItem("verify.restart", str(p), probe.status, probe.detail))
            # cores/*/*/MAS_INP_cy*.inp template decks
            decks = sorted(root.glob("cores/*/*/MAS_INP_cy*.inp"))
            if not decks:
                items.append(
                    CheckItem("verify.template", str(root / "cores"), "FAIL", "no cores/*/*/MAS_INP_cy*.inp found")
                )
            else:
                for p in decks:
                    probe = _check_template_deck(p)
                    items.append(CheckItem("verify.template", str(p), probe.status, probe.detail))
    else:
        items.append(
            CheckItem("verify.package_root", "(unset)", "SKIP", "no [verify].package_root")
        )

    # --- data source paths ---------------------------------------------------
    data_groups: list[tuple[str, list[str]]] = [
        ("data.sources", cfg.data.sources),
        ("data.lp_cache", cfg.data.lp_cache),
        ("data.lp_case_decks", cfg.data.lp_case_decks),
        ("data.eqlp_ws", cfg.data.eqlp_ws),
        ("data.ga_manifests", cfg.data.ga_manifests),
        ("data.ga_event_logs", cfg.data.ga_event_logs),
    ]
    any_data = any(paths for _, paths in data_groups)
    if not any_data:
        items.append(CheckItem("data", "(unset)", "SKIP", "no [data] source paths"))
    else:
        for group, paths in data_groups:
            for path_str in paths:
                p = _resolve(path_str, deck_dir)
                probe = _probe_read(p)
                items.append(CheckItem(group, str(p), probe.status, probe.detail))

    return items


def cmd_check(args: argparse.Namespace) -> int:
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    deck_dir = deck_path.resolve().parent
    items = _collect_checks(cfg, deck_dir)

    print(f"check  (deck: {deck_path})")
    print(f"case: mode={cfg.case.mode} pair={cfg.case.pair} feed={cfg.case.feed}")
    print()
    w_status = 11
    w_group = max((len(i.group) for i in items), default=5)
    print(f"{'STATUS':{w_status}s} {'GROUP':{w_group}s} ITEM / DETAIL")
    print("-" * (w_status + w_group + 40))

    n_pass = n_fail = n_skip = 0
    for it in items:
        if it.status in _FAIL_STATUSES:
            n_fail += 1
        elif it.status == "SKIP":
            n_skip += 1
        else:
            n_pass += 1
        print(f"{it.status:{w_status}s} {it.group:{w_group}s} {it.item}")
        if it.detail:
            print(f"{'':{w_status}s} {'':{w_group}s}   -> {it.detail}")

    print()
    print(f"summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    if n_fail:
        print(
            "RESULT: FAIL — one or more required assets are missing/unreadable. "
            "DEHYDRATED items require the cloud provider to hydrate the file(s)."
        )
        return 1
    print("RESULT: OK — all configured assets present and readable.")
    return 0


# --------------------------------------------------------------------------- #
# fuel-table (M1)
# --------------------------------------------------------------------------- #
def cmd_fuel_table(args: argparse.Namespace) -> int:
    """Build the physics fuel-feature table and print a per-library summary."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # Imported lazily so the M0 CLI (check / vendor-check) stays import-light.
    from .data.fuel_types import FuelLibrary, fuel_paths_from_config

    paths = fuel_paths_from_config(cfg)
    lib = FuelLibrary.build(cfg, persist=True)
    df = lib.frame

    print(f"fuel-table  (deck: {deck_path})")
    print(f"apr1400_root: {paths.apr1400_root}")
    print(f"ga80_hgc:     {paths.ga80_hgc}")
    print(f"store:        {paths.store}")
    print(f"rows: {len(df)}   libraries: {len(lib.libraries())}")
    print()
    header = f"{'LIBRARY':12s} {'N_TYPES':>7s} {'ENRICHMENT RANGE':>20s} {'FEATURE_POOR':>12s}"
    print(header)
    print("-" * len(header))
    for library_id in lib.libraries():
        sub = df[df["library_id"] == library_id]
        rng = lib.library_enrichment_range(library_id)
        rng_s = f"{rng[0]:.4f} - {rng[1]:.4f}" if rng else "(none)"
        n_poor = int(sub["feature_poor"].sum())
        print(f"{library_id:12s} {len(sub):7d} {rng_s:>20s} {n_poor:12d}")
    print()
    print(f"RESULT: OK — wrote {len(df)} rows to {paths.store}")
    return 0


# --------------------------------------------------------------------------- #
# extract (M2)
# --------------------------------------------------------------------------- #
def _print_extract_a(deck_path: Path, result: dict) -> None:
    print()
    print(f"extract  (dataset A, deck: {deck_path})")
    print(f"unique records : {result['n_records']}")
    print("library counts :")
    for lib, n in sorted(result["library_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {lib:24s} {n}")
    print(f"unresolved     : {result['unresolved']}")
    print(f"cbc_max coverage: {result['cbc_max_filled']} "
          f"({result['cbc_coverage_pct']:.1f}%)")
    print(f"fresh footprints: {result['footprints']}")
    print(f"converged      : {result['converged']}   "
          f"non-converged: {result['nonconverged']}")
    print(f"harvest        : {result['harvest_counts']}")
    print(f"wall time      : {result['wall_s']:.1f} s")
    print(f"store          : {result['store_dir']}")
    print(f"report         : {result['report_path']}")


def _print_extract_b(deck_path: Path, result: dict) -> None:
    print()
    print(f"extract  (dataset B, deck: {deck_path})")
    print(f"records written: {result['n_records']}  "
          f"(event {result['event_recovered']} + manifest {result['manifest_records']})")
    print(f"event uniques  : {result['n_unique_labels']}  "
          f"(raw entries {result['total_entries']})")
    print(f"pattern recovery: {result['event_recovered']} recovered / "
          f"{result['event_unrecovered']} unrecovered ({result['recovery_pct']:.1f}%)")
    print("pair counts    :")
    for pair, n in sorted(result["pair_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {pair:24s} {n}")
    print(f"converged      : {result['converged']}   "
          f"non-converged: {result['nonconverged']}   errors: {result['error_rows']}")
    print(f"parent lineage : {result['parent_resolved']} resolved / "
          f"{result['parent_unresolved']} unresolved")
    if result["audit_K1_K2"]:
        au = result["audit_K1_K2"]
        print(f"audit 20260713_061541/K1_K2: {au['entries']} labels / "
              f"{au['feasible']} feasible / {au['errors']} errors (truth 600/70/17)")
    print(f"manifests      : {result['manifest_stats']}")
    print(f"store total    : {result['records_written_total']} rows "
          f"(+{result['records_new']} new B rows submitted)")
    print(f"wall time      : {result['wall_s']:.1f} s")
    print(f"report         : {result['report_path']}")


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract a unified store from the 2_LP/eqlp caches (A) and/or 3_GA (B)."""
    dataset = str(args.dataset).lower()
    if dataset not in ("a", "b", "all"):
        print(f"lpopt extract: unknown --dataset {args.dataset!r} (expected a | b | all)")
        return 2

    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if dataset in ("a", "all"):
        from .data.extract_a import run_extract_a          # lazy: M0 stays light
        result_a = run_extract_a(
            cfg, limit=args.limit, workers=args.workers, progress=True
        )
        _print_extract_a(deck_path, result_a)

    if dataset in ("b", "all"):
        from .data.extract_b import run_extract_b          # lazy: M0 stays light
        result_b = run_extract_b(cfg, limit=args.limit, progress=True)
        _print_extract_b(deck_path, result_b)

    print("RESULT: OK")
    return 0


# --------------------------------------------------------------------------- #
# produce (M2.5)
# --------------------------------------------------------------------------- #
def cmd_produce(args: argparse.Namespace) -> int:
    """Run the stratified learning-data production campaign (plan section 5)."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not cfg.produce.strata:
        print("[ERROR] no [[produce.strata]] configured in the deck")
        return 1

    # Imported lazily so the M0 CLI (check / vendor-check) stays import-light.
    from .search.produce import run_produce

    dry_run = bool(args.dry_run)
    if not dry_run and not cfg.master.executable:
        print(
            "[ERROR] a live produce run needs [master].executable; "
            "use --dry-run for the StubEvaluator path (no MASTER)"
        )
        return 1

    # Load the physics fuel table from the produce store (when the parquet is
    # present) so the CaseAssetResolver can reach its level-3 nearest-e_core
    # restart fallback and stamp e_core/e_split on every produced row.  A
    # cross-family band pair (e.g. an auto-selected ``A2_A8`` whose package folder
    # is ``A8_A2``) has NO exact/same-pair native restart, so without the fuel
    # library its restart is unresolvable and every chain errors out.  This is why
    # a multi-PC produce kit ships ``data/store/fuel_types.parquet``.  Best-effort:
    # a missing/unreadable table simply leaves the library None (prior behaviour).
    fuel_library = None
    try:
        from .data.fuel_types import FuelLibrary
        store_dir = _resolve(cfg.produce.store_dir, deck_path.resolve().parent)
        fpath = store_dir / "fuel_types.parquet"
        if fpath.exists():
            fuel_library = FuelLibrary.from_parquet(fpath)
    except Exception as exc:  # noqa: BLE001 — never let fuel loading break produce
        print(f"[WARNING] fuel table not loaded ({exc}); "
              "e_core/level-3 restart fallback disabled")
        fuel_library = None

    summary = run_produce(
        cfg,
        dry_run=dry_run,
        max_chains=args.max_chains,
        progress=True,
        fuel_library=fuel_library,
    )
    print()
    if summary.chains == 0:
        print("RESULT: OK — nothing to produce (all strata already met)")
    else:
        print(
            f"RESULT: OK — {summary.chains} chains "
            f"({summary.converged} converged / {summary.nonconverged} nonconverged / "
            f"{summary.errors} errors), {summary.duplicates} dedup skips"
        )
        # The exploit arm silently becoming a random arm is invisible in the
        # ledger (those draws ARE random), so it is called out on the result line.
        if summary.elite_fallback_random:
            print(
                f"[WARNING] elite_perturb degraded to random on "
                f"{summary.elite_fallback_random} draw(s) — no store row qualified "
                f"as an elite parent under [produce] elite_objective="
                f"{cfg.produce.elite_objective!r}; that share of the mix was spent "
                f"on random draws."
            )
    return 0


# --------------------------------------------------------------------------- #
# boundary-probe (plan Task B — F_r boundary micro-verification)
# --------------------------------------------------------------------------- #
def cmd_boundary_probe(args: argparse.Namespace) -> int:
    """Generate + rank (+ optionally --verify) one cell's near-1.55 F_r candidates."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.verify and not cfg.master.executable:
        print("[ERROR] a --verify run needs [master].executable; omit --verify to "
              "only rank + write the candidate report (no MASTER)")
        return 1

    # Load the physics fuel table (when present) so a --verify run's resolver can
    # reach its e_core / nearest-restart fallback — same best-effort load as produce.
    fuel_library = None
    try:
        from .data.fuel_types import FuelLibrary
        store_dir = _resolve(cfg.model.store_dir, deck_path.resolve().parent)
        fpath = store_dir / "fuel_types.parquet"
        if fpath.exists():
            fuel_library = FuelLibrary.from_parquet(fpath)
    except Exception as exc:  # noqa: BLE001 — never let fuel loading break the probe
        print(f"[WARNING] fuel table not loaded ({exc}); e_core/restart fallback disabled")

    from .search.boundary_probe import run_boundary_probe

    report = run_boundary_probe(
        cfg, args.cell, top_k=args.top_k, verify=bool(args.verify),
        pool_size=args.pool_size, risk_z=args.risk_z, seed=args.seed,
        fuel_library=fuel_library, log=lambda m: print(m),
    )
    dist = report.get("top_k_f_r_mean_distribution", {})
    print()
    print(f"RESULT: OK — cell {report['cell']} "
          f"({'learned' if report['learned'] else 'UNLEARNED/quarantined'}); "
          f"{report['n_candidates']} candidates -> top {report['top_k']}")
    if dist:
        print(f"  top-{report['top_k']} predicted F_r mean: min={dist['min']:.4f} "
              f"median={dist['median']:.4f} max={dist['max']:.4f}")
    if report.get("verified"):
        v = report["verified"]
        print(f"  verified: {v['converged']} converged / {v['nonconverged']} nonconv "
              f"/ {v['error']} error -> {v.get('store_dir')}")
    print(f"  report: {report.get('_report_path')}")
    return 0


# --------------------------------------------------------------------------- #
# train / eval / optimize / report / remote (M3 / M4)
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    """Delegate to ``lpopt.model.train`` (PosValNet ensemble training)."""
    from .model.train import main as train_main
    return int(train_main(list(args.train_args)))


def cmd_v5_experiment(args: argparse.Namespace) -> int:
    """Delegate to ``lpopt.model.v5_experiment`` (pre-registered v5 integrated A/B)."""
    from .model.v5_experiment import run_from_args
    return int(run_from_args(args))


def cmd_remote(args: argparse.Namespace) -> int:
    """Delegate to ``lpopt.remote`` (gpu2-6000 push/train/status/pull/env-check)."""
    from .remote import main as remote_main
    return int(remote_main(list(args.remote_args)))


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate the trained ensemble over the split holdouts (evaluate.eval_report)."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    deck_dir = deck_path.resolve().parent
    model_dir = _resolve(args.model_dir or cfg.model.model_dir, deck_dir)
    members = sorted(model_dir.glob("member_*"))
    if not members:
        print(f"[ERROR] no member_* checkpoints under {model_dir}")
        return 1
    from .model.evaluate import eval_report
    store_dir = _resolve(cfg.model.store_dir, deck_dir)
    splits_dir = _resolve(args.splits_dir, deck_dir)
    reports_dir = _resolve(args.reports_dir, deck_dir)
    out = _resolve(args.out, deck_dir)
    result = eval_report(
        members, splits=tuple(args.splits), out=out,
        store_dir=store_dir, splits_dir=splits_dir, reports_dir=reports_dir,
        device=cfg.model.device,
    )
    print(f"\neval report -> {result['out']}")
    for v in result["verdicts"]:
        print(f"  {v['verdict']:7s} {v['criterion']} -> {v['value']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate ``report.md`` + figures for an existing ``runs/<ts>``."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[ERROR] run dir not found: {run_dir}")
        return 1
    cfg = None
    if args.input and Path(args.input).exists():
        try:
            cfg = load_config(Path(args.input))
        except ConfigError as exc:
            print(f"[WARNING] deck not loaded ({exc}); using report defaults")
    from .report.report import regenerate_report
    path = regenerate_report(run_dir, cfg, log=lambda m: print(m))
    print(f"RESULT: OK — wrote {path}")
    return 0


def cmd_sdm_mtc(args: argparse.Namespace) -> int:
    """SDM/MTC post-verification of a campaign run's top-K feasible candidates.

    Synthesises the MTC branch decks from each candidate's own final-cycle deck +
    converged restart (~1 MASTER call each), computes MTC (pcm/°C) with the
    audit-corrected scaling, writes a verdict table into the run's report dir, and
    appends the licensing labels to the ``data/sdm_mtc/results.jsonl`` sidecar.
    SDM requires a full-core rod model (not available from the quarter-core deck);
    it is reported INCONCLUSIVE here and run via the programmatic
    ``post_verify_topk(rod_model=...)`` hook (plan 12.5).
    """
    from .search.sdm_mtc import (
        BranchParams, LicensingLimits, candidates_from_delivery,
        run_post_verification, select_topk_feasible, write_verdict_table,
    )

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"[ERROR] run dir not found: {run_dir}")
        return 1
    # accept a candidates path as well as a run dir
    if run_dir.name == "candidates":
        run_dir = run_dir.parent

    cfg = None
    try:
        cfg = load_config(Path(args.input))
    except ConfigError as exc:
        print(f"[WARNING] deck not loaded ({exc}); using sdm_mtc defaults")
    sc = cfg.sdm_mtc if cfg is not None else None
    uc = cfg.constraints if cfg is not None else None

    # Decision D9: the user's ``[constraints]`` limits decide whether a measured
    # value becomes a VERDICT.  Unset -> report-only; the DCD constants stay
    # visible for context but never judge on the user's behalf.  An explicit
    # ``--mtc-limit`` / ``--sdm-limit`` is a deliberate act, so it DOES gate.
    limits = LicensingLimits.from_constraints(uc, sc)
    if args.mtc_limit is not None:
        limits.mtc_max_pcm_per_c = float(args.mtc_limit)
        limits.mtc_gated = True
    if args.sdm_limit is not None:
        limits.sdm_required_pcm = float(args.sdm_limit)
        limits.sdm_gated = True
    if args.mtc_limit is not None or args.sdm_limit is not None:
        limits.limits_source = "cli_override"
    if not (limits.mtc_gated or limits.sdm_gated):
        print("[NOTE] no user limit set ([constraints] mtc_max_pcm_per_c / "
              "sdm_required_pcm, or --mtc-limit / --sdm-limit) — running "
              "REPORT-ONLY: values are measured and recorded, nothing is marked "
              "a violator.")
    top_k = args.top_k if args.top_k is not None else (
        (uc.post_verify_top_k if uc and uc.post_verify_top_k else None)
        or (sc.top_k if sc else 5)
    )
    mtc_params = BranchParams(
        mtc_delta_c=(sc.mtc_delta_c if sc else 5.0),
        mtc_output_units=(sc.mtc_output_units if sc else "pcm_per_c"),
    )

    # D9 target set = the DELIVERY ranking (flat band, feasible EXCLUDING F_r)
    # when the run produced one; otherwise the legacy archive-feasible top-K.
    candidates, skipped = candidates_from_delivery(run_dir, None, top_k)
    for entry in skipped:
        print(f"[WARNING] skipped {entry.get('record_id')}: {entry.get('reason')}")
    if not candidates and not skipped:
        candidates = select_topk_feasible(run_dir, top_k)
    if not candidates:
        print(f"[ERROR] no candidate with a resolvable deck + its OWN converged "
              f"restart under {run_dir} (delivery.json ranking / candidates dir)")
        return 1

    if cfg is None or not cfg.master.executable or not cfg.verify.package_root:
        print("[ERROR] a live sdm-mtc run needs [master].executable + "
              "[verify].package_root in the deck")
        return 1
    base = Path(args.input).resolve().parent
    master_cfg = {
        "executable": cfg.master.executable,
        "package_root": str(_resolve(cfg.verify.package_root, base)),
        "timeout": (sc.branch_timeout_s if sc else 300.0),
    }
    results = run_post_verification(
        candidates, limits, master_cfg, run_dir / "sdm_mtc",
        mtc_params=mtc_params,
        sidecar_path=(sc.sidecar_path if sc else "data/sdm_mtc/results.jsonl"),
    )
    out = write_verdict_table(results, run_dir, limits)
    n_pass = sum(1 for r in results if r.verdict == "PASS")
    n_bad = sum(1 for r in results if r.violates)
    calls = sum(r.master_calls for r in results)
    print(f"RESULT: OK — verified {len(results)} candidate(s) "
          f"({n_pass} PASS, {n_bad} VIOLATOR) in {calls} MASTER branch call(s) "
          f"-> {out}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run the guided-search campaign (plan sec. 4.6).  ``--dry-run`` -> StubEvaluator."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    try:
        cfg.case.validate()
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    deck_dir = deck_path.resolve().parent
    dry_run = bool(args.dry_run)
    if not dry_run and not cfg.master.executable:
        print(
            "[ERROR] a live campaign needs [master].executable; use --dry-run for "
            "the StubEvaluator path (no MASTER)."
        )
        return 1

    # -- model backend --------------------------------------------------------
    store_dir = _resolve(cfg.model.store_dir, deck_dir)
    model_dir = _resolve(cfg.model.model_dir, deck_dir)
    device = cfg.model.device
    backend_factory = None
    if cfg.model.backend == "sklearn_fallback":
        from .model.model_sklearn import SklearnBackend
        from .vendor.masterrl.domain import CaseKey
        model = SklearnBackend.fit_from_store(
            store_dir, [CaseKey(str(cfg.case.pair), int(cfg.case.feed))],
            library_id=cfg.model.library_id,
        )
    else:
        from .model.model_api import PosValCnnBackend
        if not model_dir.exists():
            print(f"[ERROR] model dir not found: {model_dir}")
            return 1
        model = PosValCnnBackend.from_dir(
            model_dir, store_dir=store_dir, library_id=cfg.model.library_id, device=device
        )
        backend_factory = lambda ckpt: PosValCnnBackend.from_dir(  # noqa: E731
            ckpt, store_dir=store_dir, library_id=cfg.model.library_id, device=device
        )

    # -- evaluator ------------------------------------------------------------
    if dry_run:
        from .search.stub import StubEvaluator
        stub = StubEvaluator()
        evaluator_factory = lambda worker_id, cpu_core: stub  # noqa: E731
    else:
        evaluator_factory = None  # live default MASTER (needs package + exe)

    from .search.campaign import run_campaign
    try:
        result = run_campaign(
            cfg, model, evaluator_factory,
            dry_run=dry_run, budget=args.budget, run_dir=args.run_dir,
            resume=bool(args.resume), max_waves=args.max_waves,
            backend_factory=backend_factory, progress=True,
            early_stop=not bool(args.no_early_stop),
        )
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}")
        return 2

    print()
    if result.status == "proposals_only":
        print(f"RESULT: OK — proposals-only ({len(result.proposals)} candidates) "
              f"at {result.run_dir}")
    else:
        best = result.best
        objective = getattr(cfg.acquisition, "objective", "target_cycle")
        if objective == "min_fr_max_cycle":
            if best:
                best_txt = (
                    f"best FEASIBLE F_r {best['f_r']:.3f} (<= 1.55) @ cyclen "
                    f"{best['cyclen']:.1f} EFPD"
                )
            else:
                ov = getattr(result, "best_overall", None)
                if ov:
                    best_txt = (
                        f"NO feasible LP (F_r <= 1.55 not reached in {result.budget_spent} "
                        f"calls); best-overall F_r {ov['f_r']:.3f} "
                        f"(margin to 1.55: {ov.get('f_r_margin_to_limit'):+.3f}) @ cyclen "
                        f"{ov['cyclen']:.1f} EFPD"
                    )
                else:
                    best_txt = "no converged LP found"
        elif not best:
            best_txt = "no feasible LP found"
        elif objective == "max_cycle_min_fr":
            best_txt = (
                f"best cyclen {best['cyclen']:.1f} EFPD, F_r {best['f_r']:.3f} "
                f"(obj cyclen-λ·F_r={best['objective']:.1f})"
            )
        elif best.get("distance") is None:
            # Every objective that RETIRED the 625-EFPD target sets ``distance``
            # to None by contract (``CampaignDriver._best_dict``: flat_power,
            # fr_boundary, min_fuel_cost, …), so the target-cycle line below
            # cannot render one — it raised ``TypeError: unsupported format
            # string passed to NoneType.__format__`` *after* a finished campaign
            # had already written every artefact.  Report each mode in its own
            # units instead, defensively: a record-only / unlabelled field is
            # printed as "n/a" rather than sinking the summary of a 100-call run.
            def _f(key: str, fmt: str = ".3f") -> str:
                value = best.get(key)
                try:
                    return format(float(value), fmt)
                except (TypeError, ValueError):
                    return "n/a"
            if objective == "flat_power":
                best_txt = (
                    f"best FLAT node_peak {_f('node_peak', '.4f')} / map_cov "
                    f"{_f('map_cov', '.4f')} (flatness objective "
                    f"{_f('objective', '.4f')}) @ cyclen {_f('cyclen', '.1f')} "
                    f"EFPD [record-only], F_r {_f('f_r')} (gate "
                    f"{_f('f_r_limit_applied')}), F_q {_f('f_q')}, CBC "
                    f"{_f('cbc_max', '.0f')}"
                )
            else:
                best_txt = (
                    f"best objective {_f('objective')} @ cyclen "
                    f"{_f('cyclen', '.1f')} EFPD, F_r {_f('f_r')}"
                )
        else:
            best_txt = (
                f"best cyclen {best['cyclen']:.1f} EFPD (|Δ625|={best['distance']:.1f}), "
                f"F_r {best['f_r']:.3f}"
            )
        print(
            f"RESULT: {result.status} — {result.waves} waves, "
            f"budget {result.budget_spent}/{result.budget}, "
            f"{result.n_feasible} feasible / {result.on_target} on-target; {best_txt}"
        )
        print(f"run dir: {result.run_dir}")
    return 0


# --------------------------------------------------------------------------- #
# fuelcost-search (outer cell race for the minimum-fuel-cost configuration)
# --------------------------------------------------------------------------- #
def cmd_fuelcost_search(args: argparse.Namespace) -> int:
    """Outer cell-race search for the minimum-fuel-cost 625-EFPD configuration.

    Enumerates (e_core band, feed) cells over the search space, pre-ranks them by
    store-empirical cyclen-band feasibility + FE prior, wave-0 screens the top
    cells with the champion (free), then races a per-cell ``min_fuel_cost``
    CampaignDriver with deterministic FE-dominance elimination."""
    import json as _json

    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    deck_dir = deck_path.resolve().parent
    if not bool(args.dry_run) and not cfg.master.executable:
        print("[ERROR] a live search needs [master].executable; use --dry-run "
              "for the StubEvaluator path.")
        return 1

    store_dir = _resolve(cfg.model.store_dir, deck_dir)
    model_dir = _resolve(cfg.model.model_dir, deck_dir)
    from .model.model_api import PosValCnnBackend
    if not model_dir.exists():
        print(f"[ERROR] model dir not found: {model_dir}")
        return 1
    model = PosValCnnBackend.from_dir(
        model_dir, store_dir=store_dir, library_id=cfg.model.library_id,
        device=cfg.model.device,
    )

    from .data.fuel_types import FuelLibrary
    fuel_path = _resolve("data/store/fuel_types.parquet", deck_dir)
    fuel = FuelLibrary.from_parquet(fuel_path) if fuel_path.exists() else None
    store_df = None
    try:
        from .data.store import StoreReader
        store_df = StoreReader(store_dir).records
    except Exception as exc:  # noqa: BLE001 — store is an optional prior
        print(f"[warn] store prior unavailable ({exc}); cells stay band-neutral")

    from .search.fuelcost_search import FuelCostOuterSearch, restart_bearing_pairs
    e_targets = [float(x) for x in str(args.e_core_targets).split(",")]
    feeds = [int(x) for x in str(args.feeds).split(",")]

    # Restrict to pairs with a NATIVE MASTER restart base — a cell whose pair has
    # no exact/same-pair restart would fall back to an incompatible cross-pair
    # restart (burnt types absent from the ga80 %LPD_B&C) and die at INITIALIZE,
    # silently burning budget (forensic 20260721).  --all-pairs disables the guard.
    restart_pairs = None
    if not bool(getattr(args, "all_pairs", False)):
        pkg = _resolve(cfg.verify.package_root, deck_dir)
        restart_pairs = restart_bearing_pairs(pkg)
        print(f"[fuelcost] {len(restart_pairs)} restart-bearing pairs in {pkg}/bases: "
              f"{sorted(restart_pairs)}")
        if not restart_pairs:
            print(f"[ERROR] no restart bases under {pkg}/bases — cannot verify any cell; "
                  f"check [verify].package_root (or pass --all-pairs for a stub dry-run).")
            return 1

    dry_run = bool(args.dry_run)
    stub = None
    if dry_run:
        from .search.stub import StubEvaluator
        stub = StubEvaluator()

    def driver_factory(cell, budget, run_dir):
        from .search.campaign import CampaignDriver
        from dataclasses import replace as _rp
        case = _rp(cfg.case, mode="fixed", pair=cell.pair, feed=cell.feed)
        acq = _rp(cfg.acquisition, objective="min_fuel_cost")
        cell_cfg = _rp(cfg, case=case, acquisition=acq)
        ev = (lambda w, c: stub) if dry_run else None
        return CampaignDriver(
            cell_cfg, model, ev, dry_run=dry_run, budget=budget, run_dir=run_dir,
            resume=True, fuel_library=fuel, progress=False,
        )

    search = FuelCostOuterSearch(
        cfg, model, e_core_targets=e_targets, feeds=feeds,
        e_core_tol=float(args.e_core_tol), screen_top_k=int(args.screen_top_k),
        mini_wave=int(args.mini_wave), total_budget=int(args.budget),
        fuel_library=fuel, store_df=store_df, restart_pairs=restart_pairs,
        driver_factory=driver_factory,
    )
    run_root = _resolve(args.run_dir or (Path(cfg.flow.output_root) /
                        f"fuelcost_{time.strftime('%Y%m%d_%H%M%S')}"), deck_dir)
    result = search.run(run_root)

    summary = {
        "best_cell": result.best_cell, "best_fe": result.best_fe,
        "best": result.best, "cells_screened": result.cells_screened,
        "cells_raced": result.cells_raced, "cells_eliminated": result.cells_eliminated,
        "budget_spent": result.budget_spent, "per_cell": result.per_cell,
    }
    (run_root / "fuelcost_search.json").write_text(
        _json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print()
    if result.best is None:
        print(f"RESULT: no feasible min-fuel-cost cell found in {result.budget_spent} "
              f"MASTER calls ({result.cells_raced} cells raced)")
    else:
        b = result.best
        print(f"RESULT: min fuel cost = cell {result.best_cell}, FE={result.best_fe:.2f}, "
              f"F_r {b.get('f_r')}, cyclen {b.get('cyclen')} EFPD "
              f"({result.cells_raced} raced, {result.cells_eliminated} FE-eliminated, "
              f"budget {result.budget_spent})")
    print(f"run dir: {run_root}")
    return 0


# --------------------------------------------------------------------------- #
# curriculum (plan section 12.2/12.3 — cell-sequential curriculum)
# --------------------------------------------------------------------------- #
def cmd_curriculum(args: argparse.Namespace) -> int:
    """Run the cell-sequential curriculum driver (blind probe -> produce ->
    retrain -> validate gate, per cell, outward from the support anchor)."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    dry_run = bool(args.dry_run)
    if not dry_run and not cfg.master.executable:
        print("[ERROR] a live curriculum run needs [master].executable; "
              "use --dry-run for the StubEvaluator path (no MASTER)")
        return 1

    from .curriculum import CurriculumDriver
    driver = CurriculumDriver(cfg, dry_run=dry_run, progress=True)
    result = driver.run(max_cells=args.max_cells, resume=bool(args.resume))

    print()
    print(f"curriculum: status={result['status']}  cursor={result.get('cursor')}")
    for c in result.get("cells", []):
        print(f"  {c['cell']:20s} -> {c['outcome']:8s} (phase {c['phase']})")
    if result["status"] in ("pending", "fail"):
        print(f"\nRESUME: {result['resume_cmd']}")
        # 'pending' (produce launched / running) is not an error exit; 'fail' is.
        return 0 if result["status"] == "pending" else 1
    print(f"\nRESULT: OK — {result['status']}")
    return 0


def cmd_curriculum_produce(args: argparse.Namespace) -> int:
    """Hidden: run one cell's production (spawned detached by the driver)."""
    from .curriculum import cmd_curriculum_produce as _impl
    return int(_impl(args))


# --------------------------------------------------------------------------- #
# multi-PC produce kit (export a portable produce folder / merge results back)
# --------------------------------------------------------------------------- #
def cmd_export_produce_kit(args: argparse.Namespace) -> int:
    """Build a portable produce kit for assigned curriculum cells (a SECOND PC
    runs ``lpopt produce`` on it, then the results merge back)."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    cells = [c.strip() for c in str(args.cells).split(",") if c.strip()]
    if not cells:
        print("[ERROR] --cells is empty (use --cells 5.25-5.5_f101,5-5.25_f125)")
        return 1

    from .multi_pc import KitError, export_produce_kit

    try:
        result = export_produce_kit(
            cfg, cells, args.out, n_target=args.n_target, log=lambda m: print(m)
        )
    except KitError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print()
    print(f"RESULT: OK — kit for {len(result.cells)} cell(s) at {result.out_dir}")
    print(f"  deck   : {result.deck_path.name}  (edit [master].executable + "
          "[verify].package_root on the 2nd PC)")
    print(f"  run    : python -m lpopt produce --input {result.deck_path.name}")
    print(f"  readme : {result.readme_path.name}")
    return 0


def cmd_merge_store(args: argparse.Namespace) -> int:
    """Merge a returned produce-kit ``data/`` folder into the main store + ledger."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    from .multi_pc import KitError, merge_store

    try:
        report = merge_store(cfg, args.from_dir, dry_run=bool(args.dry_run),
                             store_dir=args.store_dir, ledger=args.ledger,
                             log=lambda m: print(m))
    except KitError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print()
    print(report.text())
    print()
    if report.dry_run:
        print("RESULT: OK — dry-run (no store/ledger written)")
    else:
        print(
            f"RESULT: OK — merged {report.new_rows} new / {report.upgraded_rows} "
            f"upgraded / {report.duplicate_rows} duplicate rows "
            f"(+{report.ledger.get('appended', 0)} ledger lines)"
        )
    return 0


# --------------------------------------------------------------------------- #
# design (plan section 12 — parametric fuel-design production chain)
# --------------------------------------------------------------------------- #
def _design_ctx(args: argparse.Namespace):
    """Load the deck and resolve the design section's paths + registry."""
    cfg = load_config(Path(args.input))
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    d = cfg.design

    def _res(p):
        p = Path(p)
        return p if p.is_absolute() else (base / p)

    store = _res(d.store_dir)
    store.mkdir(parents=True, exist_ok=True)
    from .design.spec import DesignRegistry
    registry = DesignRegistry.load(store / "registry.json")
    return cfg, d, base, store, registry, _res


def cmd_design(args: argparse.Namespace) -> int:
    print("lpopt design: choose a subcommand "
          "(generate | run | build-lib | bootstrap | pathfinder)")
    return 2


def cmd_design_generate(args: argparse.Namespace) -> int:
    """LHS-sample the design grid and write dec_FA decks."""
    try:
        cfg, d, base, store, registry, _res = _design_ctx(args)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    from .design.spec import lhs_grid
    from .design.lattice import write_dec_deck

    n = args.n if args.n is not None else d.n_types
    designs = lhs_grid(n, seed=d.seed)
    apr = _res(d.apr1400_root)
    deck_dir = store / "decks"
    for des in designs:
        write_dec_deck(des, deck_dir / registry.alias(des), registry, apr)
    registry.save(store / "registry.json")
    print(f"design generate: {len(designs)} designs -> {deck_dir}")
    for des in designs[:12]:
        print(f"  {registry.alias(des):3s}  {des.type_id}")
    if len(designs) > 12:
        print(f"  ... (+{len(designs) - 12} more)")
    return 0


def cmd_design_run(args: argparse.Namespace) -> int:
    """Run DeCART2D on generated decks (concurrent, bounded by max_parallel)."""
    try:
        cfg, d, base, store, registry, _res = _design_ctx(args)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    from .design.spec import FuelDesign
    from .design.lattice import run_batch

    manifest = store / "decks"
    # designs come from the registry (type_id -> alias); reconstruct FuelDesigns.
    from .design.spec import DESIGN_GRID  # noqa: F401  (kept for future validation)
    designs = _designs_from_registry(registry)
    if args.limit:
        designs = designs[: args.limit]
    if not designs:
        print("[ERROR] no designs in registry; run `design generate` first")
        return 1
    exe = d.decart_exe
    runs = run_batch(designs, store / "work", registry, _res(d.apr1400_root),
                     exe=exe, max_parallel=d.max_parallel, timeout_s=d.decart_timeout)
    ok = sum(1 for r in runs if r.hgc_path is not None)
    print(f"design run: {ok}/{len(runs)} DeCART runs produced HGCs")
    for r in runs:
        wall = f"{r.wall_s:.0f}s" if r.wall_s else "?"
        print(f"  {r.alias:3s}  wall={wall:>6s}  "
              f"{'OK ' + r.hgc_path.name if r.hgc_path else 'FAIL ' + (r.error or '')}")
    return 0 if ok == len(runs) else 1


def cmd_design_build_lib(args: argparse.Namespace) -> int:
    """Build the paramA MASTER library (MAS_XSL/MAS_HFF) from produced HGCs."""
    try:
        cfg, d, base, store, registry, _res = _design_ctx(args)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    from .design.library import build_master_library, default_tool_paths

    hgc_dir = _res(args.hgc_dir) if args.hgc_dir else (store / "hgc")
    hgcs = sorted(hgc_dir.glob("FA_*.HGC"))
    if not hgcs:
        print(f"[ERROR] no FA_*.HGC under {hgc_dir}")
        return 1
    out = _res(args.out) if args.out else (store / "package" / "lib")
    tp = default_tool_paths(_res(d.apr1400_root))
    if d.mas_ref:
        tp["mas_ref"] = _res(d.mas_ref)
    if d.prolog_exe:
        tp["prolog_exe"] = _res(d.prolog_exe)
    if d.totalbatcher_exe:
        tp["totalbatcher_exe"] = _res(d.totalbatcher_exe)
    build = build_master_library(hgcs, out, mas_ref=tp["mas_ref"],
                                 prolog_exe=tp["prolog_exe"],
                                 totalbatcher_exe=tp["totalbatcher_exe"])
    print(f"design build-lib: {build.comp_count} COMP + {build.refl_count} REFL "
          f"(ncomp={build.ncomp}) -> {build.xsl_path}")
    print("  sets: " + ", ".join(build.set_names))
    return 0


def cmd_design_bootstrap(args: argparse.Namespace) -> int:
    """Bootstrap one (pair, feed) band-seed restart to equilibrium."""
    try:
        cfg, d, base, store, registry, _res = _design_ctx(args)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    import random as _random
    from .design.bootstrap import make_band_restart

    pkg = _res(d.package_root) if d.package_root else (store / "package")
    exe = d.master_exe or cfg.master.executable
    if not exe:
        print("[ERROR] no MASTER executable ([design].master_exe or [master].executable)")
        return 1
    # CLI flag wins over the deck's [design].cy1_cap_efpd; None = natural-EOC cy1.
    cy1_cap = args.cy1_cap_efpd if args.cy1_cap_efpd is not None else d.cy1_cap_efpd
    if cy1_cap is not None:
        print(f"design bootstrap: cy1 capped at {float(cy1_cap):g} EFPD "
              f"(natural-EOC cy1 disabled)")
    res = make_band_restart(pkg, args.pair, args.feed, _random.Random(d.seed),
                            exe=exe, max_cycles=d.bootstrap_max_cycles,
                            enable_pin_burnup=d.enable_pin_burnup,
                            cy1_cap_efpd=cy1_cap)
    if res.error:
        print(f"[ERROR] bootstrap failed: {res.error}")
        return 1
    print(f"design bootstrap: {res.pair} feed={res.feed} folder={res.folder}")
    print(f"  converged={res.converged}  cycles_needed={res.cycles_needed}  "
          f"wall={res.wall_s:.0f}s")
    print(f"  cyclen={res.cyclen}  F_r={res.f_r}  CBC_max={res.cbc_max}  "
          f"max_pin_burnup={res.max_pin_burnup}  discharge_bu={res.discharge_burnup}")
    print(f"  restart -> {res.restart_path}")
    return 0 if res.converged else 1


def cmd_design_pathfinder(args: argparse.Namespace) -> int:
    """Run the 4-type end-to-end pathfinder acceptance gate (plan 12.1)."""
    try:
        cfg, d, base, store, registry, _res = _design_ctx(args)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    from .design.pathfinder import run_pathfinder

    exe_master = d.master_exe or cfg.master.executable
    result = run_pathfinder(cfg, store / "pathfinder", registry,
                            apr1400_root=_res(d.apr1400_root),
                            decart_exe=d.decart_exe, master_exe=exe_master,
                            max_parallel=d.max_parallel,
                            skip_decart=args.skip_decart)
    print(result.report())
    return 0 if result.ok else 1


def cmd_geom_validate(args: argparse.Namespace) -> int:
    """Pre-campaign geometry-validation protocol (review sec. 4c).

    Generates admissible pin-pitch / pin-radius variant dec decks (GEOM edit with a
    frozen assembly-envelope guard), runs DeCART (cap 4; skipped in --dry-run),
    harvests to a SIDE table, blind-probes the champion vs MASTER truth per variant,
    scores the acceptance bands, and writes a verdict report to the scratch dir.
    """
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    def _grid(spec: str) -> list[float]:
        return [float(x) / 100.0 for x in str(spec).split(",") if x.strip() != ""]

    pitch_fracs = _grid(args.pitch_grid)
    radius_fracs = _grid(args.radius_grid)
    if not pitch_fracs or not radius_fracs:
        print("[ERROR] --pitch-grid and --radius-grid must each list >=1 percent value "
              "(e.g. --pitch-grid -3,-1,0,0.5 --radius-grid -3,-1,0,1)")
        return 1

    from .design.geomcheck import DEFAULT_SCRATCH, GeomCheckError, run_geom_validation
    from .design.spec import ANCHOR_DESIGNS

    dry_run = bool(args.dry_run)
    model = None
    if not dry_run:
        # a live run scores the CURRENT champion (no fine-tune) against MASTER truth.
        deck_dir = deck_path.resolve().parent
        model_dir = _resolve(args.model_dir or cfg.model.model_dir, deck_dir)
        if not model_dir.exists():
            print(f"[ERROR] a live geom-validate needs the champion at {model_dir} "
                  "(or pass --dry-run)")
            return 1
        from .model.model_api import PosValCnnBackend
        store_dir = _resolve(cfg.model.store_dir, deck_dir)
        model = PosValCnnBackend.from_dir(model_dir, store_dir=store_dir,
                                          library_id=cfg.model.library_id,
                                          device=cfg.model.device)
        if not cfg.master.executable:
            print("[ERROR] a live geom-validate needs [master].executable for the MASTER "
                  "blind probe (or pass --dry-run)")
            return 1

    scratch = Path(args.scratch) if args.scratch else DEFAULT_SCRATCH
    anchors = list(ANCHOR_DESIGNS)[: args.anchors] if args.anchors else list(ANCHOR_DESIGNS)
    try:
        result = run_geom_validation(
            cfg, pitch_fracs=pitch_fracs, radius_fracs=radius_fracs, anchors=anchors,
            feed=args.feed, probe_size=args.probe_size, dry_run=dry_run,
            scratch_dir=scratch, model=model, seed=args.seed, log=lambda m: print(m),
        )
    except GeomCheckError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print()
    print(f"RESULT: {result.overall_verdict} — {result.n_admissible}/{result.n_variants} "
          f"admissible variants; verdict -> {result.verdict_path}")
    if result.ood_warnings:
        print(f"  OOD-flagged variants: {len(result.ood_warnings)} "
              "(geometry off the fixed-geometry training manifold, as expected)")
    return 0 if result.overall_verdict == "PASS" else 1


def _designs_from_registry(registry):
    """Reconstruct FuelDesign objects from persisted descriptive type_ids."""
    import re
    from .design.spec import FuelDesign
    out = []
    pat = re.compile(r"^P(\d{2})(\d{2})Z([12])G(\d{2})N(\d{2})$")
    for type_id in registry.mapping:
        m = pat.match(type_id)
        if not m:
            continue
        e1 = int(m.group(1)) / 10
        e2 = int(m.group(2)) / 10
        out.append(FuelDesign(e1, e2, f"z{m.group(3)}", float(int(m.group(4))),
                              int(m.group(5))))
    return out


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
def _make_stub(name: str, milestone: str):
    def _stub(_args: argparse.Namespace) -> int:
        print(f"lpopt {name}: not implemented (milestone {milestone})")
        return 2

    return _stub


# --------------------------------------------------------------------------- #
# frontier-produce (F_r=1.55 boundary training campaign — one round per invocation)
# --------------------------------------------------------------------------- #
def cmd_frontier_produce(args: argparse.Namespace) -> int:
    """Run EXACTLY ONE round of the fr_boundary boundary campaign, then exit.

    Emits the roster JSON (every grid cell with base_status + exclusion reason),
    then — for a live (non-dry-run) round — REFUSES to construct a MASTER evaluator
    unless ``LPOPT_WORKER=1`` (set only inside the PC2 kit's run_frontier.bat), so a
    local PC can never launch MASTER by accident.  Writes ``frontier_round.json`` and
    returns; the supervising session drives the AL cadence between rounds.

    SMOKE (one cell, one MASTER call) — use the OFFICIAL CLI, never an ad-hoc script:
    ``python -m lpopt optimize --input <kit-deck with [case] mode=fixed pair/feed>
    --budget 1``.  ``python -m lpopt`` carries the ``__main__`` guard (:mod:`lpopt.__main__`)
    and drives MASTER via subprocess, so it is safe under Windows process spawn; a
    bare module without that guard deadlocks worker start-up (forensic 20260723)."""
    import json as _json
    import os as _os

    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    deck_dir = deck_path.resolve().parent
    run_root = _resolve(args.run_root, deck_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    from .search.frontier_search import FrBoundaryOuterRace

    # optional per-round weight override (round>=2 boundary-proximity weights).
    weights = None
    if getattr(args, "weights", None):
        wp = _resolve(args.weights, deck_dir)
        if wp.exists():
            weights = _json.loads(wp.read_text(encoding="utf-8"))

    dry_run = bool(getattr(args, "dry_run", False))

    # Multi-PC disjoint-split: the peer PC's cell ids (this PC skips them so two
    # boxes run SEROSO rosters with per-PC seeds — no duplicate labels).
    exclude_cells = {c.strip() for c in str(getattr(args, "exclude_cells", "") or "").split(",")
                     if c.strip()} or None

    # -- LPOPT_WORKER guard: a LIVE round is structurally impossible off PC2 ---- #
    if not dry_run and _os.environ.get("LPOPT_WORKER") != "1":
        # still emit the roster so the caller sees the plan, then refuse.
        _race = FrBoundaryOuterRace(cfg, None, run_root=run_root,
                                    round_budget=int(args.round_budget),
                                    exclude_cells=exclude_cells)
        (run_root / "roster.json").write_text(
            _json.dumps(_race.roster_report(), indent=2, default=str), encoding="utf-8")
        print("[ERROR] a live frontier round runs MASTER; refusing without "
              "LPOPT_WORKER=1 (set only inside the PC2 kit run_frontier.bat). "
              "Use --dry-run for the StubEvaluator path.")
        return 1

    # -- model + per-cell driver factory --------------------------------------- #
    store_dir = _resolve(cfg.model.store_dir, deck_dir)
    model_dir = _resolve(cfg.model.model_dir, deck_dir)
    model = None
    backend_factory = None
    stub = None
    if dry_run:
        from .search.stub import StubEvaluator
        stub = StubEvaluator()

    def _load_model(mdir):
        from .model.model_api import PosValCnnBackend
        return PosValCnnBackend.from_dir(
            Path(mdir), store_dir=store_dir, library_id=cfg.model.library_id,
            device=cfg.model.device)

    if not dry_run:
        if not model_dir.exists():
            print(f"[ERROR] champion model dir not found: {model_dir}")
            return 1
        model = _load_model(model_dir)
        backend_factory = _load_model            # fresh handle per cell (reproducible)

    from .data.fuel_types import FuelLibrary
    fuel_path = _resolve("data/store/fuel_types.parquet", deck_dir)
    fuel = FuelLibrary.from_parquet(fuel_path) if fuel_path.exists() else None

    # objective is deck-driven; the outer race now ALSO reads it, because the
    # budget-allocation rule differs by objective (decision D7: flat_power
    # allocates by map-coverage deficit, fr_boundary by F_r proximity).  Passing
    # it explicitly keeps the inner campaign and the outer allocation from
    # disagreeing about which campaign is being run.
    _obj = str(getattr(cfg.acquisition, "objective", "fr_boundary") or "fr_boundary")
    if _obj not in ("fr_boundary", "flat_power", "min_fuel_cost"):
        _obj = "fr_boundary"

    def driver_factory(cell, budget, cell_run_dir, seed):
        from .search.campaign import CampaignDriver
        from dataclasses import replace as _rp
        case = _rp(cfg.case, mode="fixed", pair=cell.pair, feed=cell.feed)
        acq = _rp(cfg.acquisition, objective=_obj)
        cell_cfg = _rp(cfg, case=case, acquisition=acq)
        ev = (lambda w, c: stub) if dry_run else None
        return CampaignDriver(
            cell_cfg, model, ev, dry_run=dry_run, budget=budget, run_dir=cell_run_dir,
            resume=True, backend_factory=backend_factory, early_stop=False,
            seed=seed, fuel_library=fuel, progress=False,
        )

    race = FrBoundaryOuterRace(
        cfg, model, run_root=run_root, round_budget=int(args.round_budget),
        driver_factory=driver_factory, backend_factory=backend_factory,
        exclude_cells=exclude_cells, objective=_obj,
    )
    (run_root / "roster.json").write_text(
        _json.dumps(race.roster_report(), indent=2, default=str), encoding="utf-8")

    result = race.run_round()                     # exactly ONE round, then exit
    payload = {
        "round_index": result.round_index, "round_spent": result.round_spent,
        "per_cell": result.per_cell, "excluded": result.excluded,
        "retrain_events": result.retrain_events,
    }
    (run_root / "frontier_round.json").write_text(
        _json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"RESULT: frontier round {result.round_index} spent {result.round_spent} "
          f"MASTER calls over {len(result.per_cell)} cells "
          f"({len(result.excluded)} excluded). run root: {run_root}")
    return 0


# --------------------------------------------------------------------------- #
# gate-promote (honest no-regression gate + atomic champion promotion)
# --------------------------------------------------------------------------- #
def _apply_promotion(state_path: Path, deck_path: Path, new_dir: str) -> None:
    """Atomically point the curriculum state.json champion + the deck [model].model_dir
    at ``new_dir`` (BOM-safe text edit, no PowerShell Set-Content)."""
    # 1) state.json champion_model_dir
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["champion_model_dir"] = str(new_dir)
        tmp = state_path.with_name(f"{state_path.name}.tmp-{time.time_ns()}")
        tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        tmp.replace(state_path)
    # 2) deck [model].model_dir (preserve inline comments after the value)
    text = deck_path.read_text(encoding="utf-8")
    def _sub(m: re.Match) -> str:
        return f'{m.group(1)}"{new_dir}"{m.group(3)}'
    new_text, n = re.subn(
        r'(^\s*model_dir\s*=\s*)("[^"]*"|\'[^\']*\')(.*)$',
        _sub, text, count=1, flags=re.MULTILINE)
    if n:
        tmp = deck_path.with_name(f"{deck_path.name}.tmp-{time.time_ns()}")
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(deck_path)


def cmd_gate_promote(args: argparse.Namespace) -> int:
    """Run the HONEST no-regression gate (OOS-vs-OOS per-cell Spearman) + the
    legacy high-cyclen tail gate between two champion dirs, emit the gate JSON, and
    — ONLY when BOTH pass — atomically promote ``--new`` into the curriculum
    state.json champion + the deck [model].model_dir.  Replaces the nonexistent
    gate_promote.py scratch script (a committed, tested path in the AL loop)."""
    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    deck_dir = deck_path.resolve().parent
    prev_dir = _resolve(args.prev, deck_dir)
    new_dir = _resolve(args.new, deck_dir)
    store_dir = _resolve(cfg.model.store_dir, deck_dir)
    out_path = _resolve(args.out or (deck_dir / "gate.json"), deck_dir)

    from .curriculum import gate_no_regression, gate_legacy_tail
    from .model.model_api import PosValCnnBackend
    from .data.store import StoreReader

    def _load(mdir):
        return PosValCnnBackend.from_dir(
            Path(mdir), store_dir=store_dir, library_id=cfg.model.library_id,
            device=cfg.model.device)

    old_model, new_model = _load(prev_dir), _load(new_dir)
    df = StoreReader(store_dir).records
    curr = cfg.curriculum

    # per-cell OOS holdout groups from the stable-hash curriculum split.
    from .curriculum import CurriculumDriver
    drv = CurriculumDriver(cfg, model_loader=_load)
    drv._load_state()  # populate drv.state (done cells) from the real curriculum state.json
    manifest = drv._curriculum_split_manifest(records=df)
    val_by_cell = manifest.groups.get("curriculum_val_by_cell", {})
    done_cells = [c for c in drv.state.get("order", [])
                  if drv.state["cells"].get(c, {}).get("phase") == "done"]

    noreg = gate_no_regression(
        old_model, new_model, df, val_by_cell, done_cells,
        epsilon=curr.gate_noreg_epsilon,
        fr_guarded=bool(getattr(curr, "gate_noreg_fr_guard_enabled", False)))
    tail = gate_legacy_tail(old_model, new_model, df, bands=curr.gate_tail_bands,
                            feed=curr.gate_tail_feed, sample_per_band=curr.gate_tail_sample,
                            seed=cfg.flow.random_seed, epsilon=curr.gate_tail_epsilon,
                            enabled=getattr(curr, "gate_tail_enabled", True))
    passed = bool(noreg.get("pass")) and bool(tail.get("pass"))
    gate = {"pass": passed, "no_regression": noreg, "legacy_tail": tail,
            "prev": str(prev_dir), "new": str(new_dir),
            "champion_model_dir": str(new_dir) if passed else str(prev_dir)}
    out_path.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

    # Surface the honesty note on the CONSOLE too, not only in the JSON: the axes
    # that were scored-but-not-enforced (and any that could not be judged at all)
    # are exactly what a "PASS" would otherwise be silently read as covering.
    if noreg.get("note"):
        print(f"[no-regression] {noreg['note']}")
    # The RESULT line is what gets pasted into a report, so the caveat rides on it
    # rather than only two lines above.  Warn, never block (user decision
    # 2026-07-26): promotion proceeds, but "gate PASS" cannot be quoted bare while
    # a declared guard was measured in no cell at all.
    _blind = list(noreg.get("blind_targets") or [])
    _caveat = f" [GUARDS NOT MEASURED: {', '.join(_blind)}]" if _blind else ""

    if passed and getattr(args, "check_only", False):
        # Inspection path: the gates ran and the JSON is written, but NOTHING is
        # promoted.  Added 2026-07-25 after a "just look at the gate" invocation
        # swapped the champion on the spot — a command whose only mode is
        # irreversible is a foot-gun, not a safety feature.
        print("RESULT: gate PASS -> NOT promoted (--check-only); "
              "re-run without --check-only to promote" + _caveat)
    elif passed:
        # the REAL curriculum state the driver loaded (data/curriculum), not runs/.
        state_path = drv.state_path
        if args.state:
            state_path = _resolve(args.state, deck_dir)
        # write a relative, forward-slash path: an absolute Windows path (backslashes)
        # would corrupt the TOML deck string.  Fall back to a slash-normalized abs path.
        promote_val = args.new if not Path(args.new).is_absolute() else str(new_dir).replace("\\", "/")
        _apply_promotion(state_path, deck_path, promote_val)
        print(f"RESULT: gate PASS -> champion promoted to {promote_val}{_caveat}")
    else:
        print(f"RESULT: gate FAIL -> champion unchanged (no_regression="
              f"{noreg.get('pass')}, legacy_tail={tail.get('pass')})")
    print(f"gate JSON: {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# compliance-audit (R1-R3 assembly-design compliance flags)
# --------------------------------------------------------------------------- #
def cmd_compliance_audit(args: argparse.Namespace) -> int:
    """Audit the fuel library for R1/R2 compliance (octant pin symmetry + the
    enr_zone=0.85*enr_main zoning ratio) and write a sidecar report JSON.

    Types with a preserved HGC %DIST pin map are octant-checked; the all-NaN ga80
    enrichment rows audit as ``zone_ratio='unknown'`` — kept for TRAINING data (this
    campaign produces labels, not final designs) but flagged 'unknown' for any
    real-design roster reporting."""
    from .data import compliance as comp
    from .data.fuel_types import FuelLibrary

    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    deck_dir = deck_path.resolve().parent
    fuel_path = _resolve(args.fuel or "data/store/fuel_types.parquet", deck_dir)
    if not fuel_path.exists():
        print(f"[ERROR] fuel table not found: {fuel_path}")
        return 1
    fuel = FuelLibrary.from_parquet(fuel_path)

    # optional %DIST map source: a JSON of {type_id: flat-256 map} the caller built
    # from preserved HGCs (kept out-of-band so the audit needs no live HGC files).
    hgc_maps = None
    if getattr(args, "hgc_maps", None):
        mp = _resolve(args.hgc_maps, deck_dir)
        if mp.exists():
            hgc_maps = json.loads(mp.read_text(encoding="utf-8"))

    rows = comp.audit_fuel_types(fuel, library_id=args.library, hgc_maps=hgc_maps)
    from collections import Counter
    oct_c = Counter(r.octant_symmetry for r in rows)
    zone_c = Counter(r.zone_ratio for r in rows)
    report = {
        "library_id": args.library, "n_types": len(rows),
        "octant_symmetry": dict(oct_c), "zone_ratio": dict(zone_c),
        "types": [r.as_dict() for r in rows],
    }
    out_path = _resolve(args.out or (deck_dir / "compliance_audit.json"), deck_dir)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"RESULT: audited {len(rows)} {args.library} types "
          f"(octant {dict(oct_c)}, zone {dict(zone_c)}). report: {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# debug-panel (MASTER-verified scoring panel, neutronics-unit tolerances)
# --------------------------------------------------------------------------- #
def cmd_debug_panel(args: argparse.Namespace) -> int:
    print("lpopt debug-panel: choose a subcommand (score)")
    return 2


def cmd_debug_panel_score(args: argparse.Namespace) -> int:
    """Score a champion against the MASTER-verified debug panel (REPORT ONLY).

    ALWAYS returns 0, including when a target blows its tolerance and when the
    panel is empty.  This command exists to be run automatically after every
    build (user directive 2026-07-29); a report that can fail a build is a report
    whose thresholds get negotiated away.  Deviations are printed and written to
    the JSON artifact — that is the whole enforcement mechanism, on purpose.
    """
    from .tools import debug_panel as dp

    deck_path = Path(args.input)
    try:
        cfg = load_config(deck_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 0                       # report-only: never block on a deck typo
    deck_dir = deck_path.resolve().parent
    model_dir = _resolve(args.model_dir or cfg.model.model_dir, deck_dir)
    store_dir = _resolve(cfg.model.store_dir, deck_dir)
    out_path = _resolve(args.out or (deck_dir / "debug_panel.json"), deck_dir)

    panel = cfg.debug_panel
    campaigns = ([g.strip() for g in args.campaigns.split(",") if g.strip()]
                 if args.campaigns else list(panel.campaigns))

    try:
        report = dp.run_score(
            model_dir, store_dir=store_dir, library_id=cfg.model.library_id,
            device=cfg.model.device, campaigns=campaigns,
            tolerances=panel.tolerances, out_path=out_path)
    except Exception as exc:           # noqa: BLE001 — report-only, never block
        print(f"[ERROR] debug-panel score failed: {type(exc).__name__}: {exc}")
        return 0

    print(f"debug panel: {report['n_panel_rows']} MASTER-verified rows "
          f"(campaigns {campaigns}) vs {model_dir}")
    print(dp.format_report(report))
    failed, unscored = report["failed"], report["unscored"]
    if failed:
        print(f"RESULT: OUT OF TOLERANCE: {', '.join(failed)} "
              f"(report-only — nothing is blocked)")
    elif report["passed"]:
        print("RESULT: every scored target within neutronics tolerance")
    else:
        print("RESULT: nothing scored (empty panel or no labels)")
    if unscored:
        print(f"[unscored] {', '.join(unscored)}")
    print(f"panel JSON: {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# argument parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lpopt",
        description="APR1400 equilibrium-cycle loading-pattern optimization.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_check = sub.add_parser("check", help="preflight configured assets (open-and-read 64KB)")
    p_check.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_check.set_defaults(func=cmd_check)

    p_vendor = sub.add_parser("vendor-check", help="verify vendored master_rl integrity + drift")
    p_vendor.set_defaults(func=cmd_vendor_check)

    p_fuel = sub.add_parser("fuel-table", help="build physics fuel-feature table (data/store/fuel_types.parquet)")
    p_fuel.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_fuel.set_defaults(func=cmd_fuel_table)

    p_extract = sub.add_parser("extract", help="extract unified store from 2_LP/eqlp (A) and 3_GA (B) sources")
    p_extract.add_argument("--dataset", default="a", help="a (2_LP/eqlp) | b (3_GA) | all (A then B)")
    p_extract.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_extract.add_argument("--limit", type=int, default=None, help="cap unique records (smoke runs)")
    p_extract.add_argument("--workers", type=int, default=None, help="harvest worker processes (default [extract].workers)")
    p_extract.set_defaults(func=cmd_extract)

    p_produce = sub.add_parser("produce", help="stratified MASTER learning-data production campaign")
    p_produce.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_produce.add_argument("--dry-run", action="store_true", help="StubEvaluator + run-scoped temp store (no MASTER)")
    p_produce.add_argument("--max-chains", type=int, default=None, help="cap chains evaluated this invocation")
    p_produce.set_defaults(func=cmd_produce)

    p_train = sub.add_parser("train", help="train the PosValNet ensemble (delegates to lpopt.model.train)")
    p_train.add_argument("train_args", nargs=argparse.REMAINDER, help="args forwarded to lpopt.model.train")
    p_train.set_defaults(func=cmd_train)

    p_v5 = sub.add_parser(
        "v5-experiment",
        help="pre-registered v5 integrated A/B (--dry-run validates + prints the plan)")
    # Options are registered by the implementation module itself so the two entry
    # points can never drift; the import is cheap (no torch at module scope).
    from .model.v5_experiment import add_arguments as _v5_add_arguments
    _v5_add_arguments(p_v5)
    p_v5.set_defaults(func=cmd_v5_experiment)

    p_eval = sub.add_parser("eval", help="evaluate the trained ensemble over split holdouts")
    p_eval.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_eval.add_argument("--model-dir", default=None, help="ensemble dir (default [model].model_dir)")
    p_eval.add_argument("--splits", nargs="+", default=["S0", "S1", "S2", "S4"])
    p_eval.add_argument("--out", default="data/reports/model_report.md")
    p_eval.add_argument("--splits-dir", default="data/splits")
    p_eval.add_argument("--reports-dir", default="data/reports")
    p_eval.set_defaults(func=cmd_eval)

    p_opt = sub.add_parser("optimize", help="guided-search campaign (plan sec. 4.6)")
    p_opt.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_opt.add_argument("--dry-run", action="store_true", help="StubEvaluator + run-scoped store (no MASTER)")
    p_opt.add_argument("--budget", type=int, default=None, help="MASTER budget (0 -> proposals-only)")
    p_opt.add_argument("--run-dir", default=None, help="explicit run dir (default runs/<ts>)")
    p_opt.add_argument("--resume", action="store_true", help="resume a run dir from state.json (no budget double-spend)")
    p_opt.add_argument("--max-waves", type=int, default=None, help="stop after N waves (pause; resumable)")
    p_opt.add_argument("--no-early-stop", action="store_true",
                       help="disable the on-target early-stop rule (run the full budget: 12 waves + reserve)")
    p_opt.set_defaults(func=cmd_optimize)

    p_fcs = sub.add_parser(
        "fuelcost-search",
        help="outer cell-race for the minimum-fuel-cost 625-EFPD configuration",
    )
    p_fcs.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_fcs.add_argument("--e-core-targets", default="5.0,5.125,5.25,5.375,5.5",
                       help="comma-separated e_core band centres to enumerate (>=5.0)")
    p_fcs.add_argument("--e-core-tol", type=float, default=0.125,
                       help="half-width of each e_core band")
    p_fcs.add_argument("--feeds", default="101,105,109,113,117,121,125",
                       help="comma-separated feeds (1+4N grid) to search")
    p_fcs.add_argument("--budget", type=int, default=300, help="total MASTER calls")
    p_fcs.add_argument("--screen-top-k", type=int, default=8,
                       help="cells wave-0 screened with the champion (free)")
    p_fcs.add_argument("--mini-wave", type=int, default=8,
                       help="verification calls per race mini-wave")
    p_fcs.add_argument("--run-dir", default=None, help="explicit run root (default runs/fuelcost_<ts>)")
    p_fcs.add_argument("--dry-run", action="store_true", help="StubEvaluator path (no MASTER)")
    p_fcs.add_argument("--all-pairs", action="store_true",
                       help="disable the restart-bearing-pair guard (stub dry-runs only)")
    p_fcs.set_defaults(func=cmd_fuelcost_search)

    p_report = sub.add_parser("report", help="regenerate report.md + figures for a runs/<ts>")
    p_report.add_argument("run_dir", help="the runs/<ts> directory")
    p_report.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck (for case/limits/GA log)")
    p_report.set_defaults(func=cmd_report)

    # sdm-mtc: SDM/MTC post-verification of a run's top-K feasible candidates (plan 12.5)
    p_sm = sub.add_parser("sdm-mtc", help="SDM/MTC post-verify top-K feasible candidates of a run")
    p_sm.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck (limits/executable)")
    p_sm.add_argument("--run", required=True, help="the runs/<ts> dir (or its candidates/ path)")
    p_sm.add_argument("--top-k", type=int, default=None, help="candidates to verify (default [sdm_mtc].top_k)")
    p_sm.add_argument("--mtc-limit", type=float, default=None, help="most-positive allowed MTC [pcm/C]")
    p_sm.add_argument("--sdm-limit", type=float, default=None, help="minimum required SDM [pcm]")
    p_sm.set_defaults(func=cmd_sdm_mtc)

    p_remote = sub.add_parser("remote", help="gpu2-6000 remote training (delegates to lpopt.remote)")
    p_remote.add_argument("remote_args", nargs=argparse.REMAINDER, help="args forwarded to lpopt.remote")
    p_remote.set_defaults(func=cmd_remote)

    # boundary-probe: F_r near-1.55 micro-verification harness (plan Task B)
    p_bp = sub.add_parser(
        "boundary-probe",
        help="rank (+ optionally --verify) a cell's near-1.55 F_r candidates (Task B)",
    )
    p_bp.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_bp.add_argument("--cell", required=True, help="cell id (e.g. 5-5.25_f117)")
    p_bp.add_argument("--top-k", type=int, default=16, help="candidates to select/verify")
    p_bp.add_argument("--pool-size", type=int, default=240, help="candidate pool before ranking")
    p_bp.add_argument("--risk-z", type=float, default=0.25, help="sigma multiplier for the reported UCB")
    p_bp.add_argument("--seed", type=int, default=0, help="pool RNG seed")
    p_bp.add_argument("--verify", action="store_true",
                      help="run the top-K through MASTER (produce/verify) into the store")
    p_bp.set_defaults(func=cmd_boundary_probe)

    # curriculum: cell-sequential curriculum (plan section 12.2/12.3)
    p_curr = sub.add_parser("curriculum", help="cell-sequential curriculum driver (plan sec. 12.2/12.3)")
    p_curr.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_curr.add_argument("--dry-run", action="store_true", help="StubEvaluator + run-scoped store (no MASTER)")
    p_curr.add_argument("--max-cells", type=int, default=None, help="stop after N cells this invocation")
    p_curr.add_argument("--resume", action="store_true", help="resume state.json (default; -no-resume reinitialises)")
    p_curr.set_defaults(func=cmd_curriculum, resume=True)

    # curriculum-produce: hidden detached per-cell production entry
    p_cp = sub.add_parser("curriculum-produce", help=argparse.SUPPRESS)
    p_cp.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_cp.add_argument("--cell", required=True, help="cell id (e.g. 5.25-5.5_f117)")
    p_cp.set_defaults(func=cmd_curriculum_produce)

    # export-produce-kit: build a portable produce kit for a SECOND PC
    p_kit = sub.add_parser(
        "export-produce-kit",
        help="build a portable produce kit for assigned curriculum cells (2nd PC)",
    )
    p_kit.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_kit.add_argument("--cells", required=True,
                       help="comma-separated cell ids (e.g. 5.25-5.5_f101,5-5.25_f125)")
    p_kit.add_argument("--out", required=True, help="kit output directory")
    p_kit.add_argument("--n-target", type=int, default=None,
                       help="converged-label target per cell (default [curriculum].n_target)")
    p_kit.set_defaults(func=cmd_export_produce_kit)

    # merge-store: merge a returned kit data/ folder into the main store + ledger
    p_ms = sub.add_parser(
        "merge-store", help="merge a returned produce-kit data/ folder into the main store",
    )
    p_ms.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_ms.add_argument("--from", dest="from_dir", required=True,
                      help="the returned kit data/ folder (holds store/ + produce/)")
    p_ms.add_argument("--dry-run", action="store_true",
                      help="report only; do not write the store or ledger")
    p_ms.add_argument("--store-dir", default=None,
                      help="override [model].store_dir (merge into this store instead; "
                           "relative = cwd). Use with --ledger to target a SCRATCH copy.")
    p_ms.add_argument("--ledger", default=None,
                      help="override [produce].ledger (merge into this ledger instead; "
                           "relative = cwd)")
    p_ms.set_defaults(func=cmd_merge_store)

    # design: parametric fuel-design production chain (plan section 12)
    p_design = sub.add_parser("design", help="parametric fuel-design production chain (plan sec. 12)")
    p_design.set_defaults(func=cmd_design)
    dsub = p_design.add_subparsers(dest="design_command", metavar="<subcommand>")

    p_dg = dsub.add_parser("generate", help="LHS-sample the design grid and write dec_FA decks")
    p_dg.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_dg.add_argument("--n", type=int, default=None, help="number of designs (default [design].n_types)")
    p_dg.set_defaults(func=cmd_design_generate)

    p_dr = dsub.add_parser("run", help="run DeCART2D on generated decks (concurrent)")
    p_dr.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_dr.add_argument("--limit", type=int, default=None, help="cap number of lattices run")
    p_dr.set_defaults(func=cmd_design_run)

    p_dl = dsub.add_parser("build-lib", help="build paramA MAS_XSL/MAS_HFF via TotalBatcher4")
    p_dl.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_dl.add_argument("--hgc-dir", default=None, help="dir of FA_*.HGC (default [design].store_dir/hgc)")
    p_dl.add_argument("--out", default=None, help="output lib dir (default [design].store_dir/package/lib)")
    p_dl.set_defaults(func=cmd_design_build_lib)

    p_db = dsub.add_parser("bootstrap", help="bootstrap a (pair, feed) band-seed restart")
    p_db.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_db.add_argument("--pair", required=True, help="two aliases joined by '_' (e.g. P0_P1)")
    p_db.add_argument("--feed", type=int, default=121, help="feed (1+4N grid)")
    p_db.add_argument("--cy1-cap-efpd", type=float, default=None,
                      help="cap the throwaway cy1 fresh-core cycle at N EFPD "
                           "instead of running it to natural EOC, so cy02 starts "
                           "from an equilibrium-like carryover "
                           "(principled value: 2*B1/(241/feed+1); overrides "
                           "[design].cy1_cap_efpd)")
    p_db.set_defaults(func=cmd_design_bootstrap)

    p_pf = dsub.add_parser("pathfinder", help="4-type end-to-end acceptance gate (plan 12.1)")
    p_pf.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_pf.add_argument("--skip-decart", action="store_true", help="reuse existing HGCs (skip lattice runs)")
    p_pf.set_defaults(func=cmd_design_pathfinder)

    # geom-validate: pre-campaign geometry-validation protocol (review sec. 4c)
    p_gv = sub.add_parser(
        "geom-validate",
        help="pin-pitch/pin-radius geometry-validation transfer test (review sec. 4c)",
    )
    p_gv.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_gv.add_argument("--pitch-grid", default="-3,-1,0,0.5",
                      help="comma pin-pitch percents (respect the +1.06%% ceiling; "
                           "e.g. -3,-1,0,0.5)")
    p_gv.add_argument("--radius-grid", default="-3,-1,0,1",
                      help="comma pin-radius (co-scaled) percents (e.g. -3,-1,0,1)")
    p_gv.add_argument("--anchors", type=int, default=None,
                      help="number of enrichment/Gd anchors (default: all ANCHOR_DESIGNS)")
    p_gv.add_argument("--feed", type=int, default=121, help="probe feed (default 121)")
    p_gv.add_argument("--probe-size", type=int, default=16,
                      help="MASTER chains per variant (default 16)")
    p_gv.add_argument("--dry-run", action="store_true",
                      help="no DeCART / no MASTER: stub champion + StubEvaluator "
                           "(the DeCART-less E2E)")
    p_gv.add_argument("--model-dir", default=None,
                      help="champion ensemble dir for a live run (default [model].model_dir)")
    p_gv.add_argument("--scratch", default=None,
                      help="scratch dir (default C:/Users/USER/AppData/Local/Temp/eqlp_geomchk)")
    p_gv.add_argument("--seed", type=int, default=0, help="probe RNG seed")
    p_gv.set_defaults(func=cmd_geom_validate)

    # frontier-produce: ONE round of the fr_boundary boundary campaign, then exit
    p_fp = sub.add_parser(
        "frontier-produce",
        help="run one round of the F_r=1.55 boundary training campaign (PC2 worker)",
    )
    p_fp.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_fp.add_argument("--round-budget", type=int, default=276,
                      help="real MASTER calls this round (default 276)")
    p_fp.add_argument("--exclude-cells", default=None,
                      help="comma-separated cell ids to EXCLUDE (multi-PC disjoint "
                           "split: this PC skips the peer's half; persisted in "
                           "race_state so a resumed round keeps the same split)")
    p_fp.add_argument("--run-root", default="runs/frontier",
                      help="race run root (default runs/frontier)")
    p_fp.add_argument("--weights", default=None,
                      help="optional round>=2 proximity-weight override JSON")
    p_fp.add_argument("--dry-run", action="store_true",
                      help="StubEvaluator path (no MASTER; no LPOPT_WORKER needed)")
    p_fp.set_defaults(func=cmd_frontier_produce)

    # gate-promote: honest no-regression gate + atomic champion promotion
    p_gp = sub.add_parser(
        "gate-promote",
        help="honest no-regression + legacy-tail gate; atomic champion promote on pass",
    )
    p_gp.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_gp.add_argument("--prev", required=True, help="incumbent champion model dir")
    p_gp.add_argument("--new", required=True, help="candidate champion model dir")
    p_gp.add_argument("--out", default=None, help="gate JSON output (default ./gate.json)")
    p_gp.add_argument("--state", default=None,
                      help="curriculum state.json to promote in (default runs/curriculum/state.json)")
    p_gp.add_argument("--check-only", action="store_true",
                      help="run both gates and write the JSON but NEVER promote "
                           "(without this, a PASS promotes immediately)")
    p_gp.set_defaults(func=cmd_gate_promote)

    # compliance-audit: R1-R3 assembly-design compliance flags
    p_ca = sub.add_parser(
        "compliance-audit",
        help="audit fuel types for octant symmetry + enr_zone=0.85*enr_main (R1/R2)",
    )
    p_ca.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_ca.add_argument("--library", default="ga80", help="library id to audit (default ga80)")
    p_ca.add_argument("--fuel", default=None,
                      help="fuel table parquet (default data/store/fuel_types.parquet)")
    p_ca.add_argument("--hgc-maps", default=None,
                      help="optional JSON {type_id: flat-256 %%DIST map} for octant checks")
    p_ca.add_argument("--out", default=None,
                      help="report JSON (default ./compliance_audit.json)")
    p_ca.set_defaults(func=cmd_compliance_audit)

    # debug-panel: MASTER-verified scoring panel with neutronics-unit tolerances
    p_dp = sub.add_parser(
        "debug-panel",
        help="score a champion against MASTER-verified rows in neutronics units",
    )
    p_dp.set_defaults(func=cmd_debug_panel)
    dpsub = p_dp.add_subparsers(dest="debug_panel_command", metavar="<subcommand>")

    p_dps = dpsub.add_parser(
        "score",
        help="score --model-dir against the panel (REPORT ONLY; always exits 0)",
    )
    p_dps.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    p_dps.add_argument("--model-dir", default=None,
                       help="champion dir to score (default [model].model_dir)")
    p_dps.add_argument("--campaigns", default=None,
                       help="comma-separated campaign globs overriding "
                            "[debug_panel].campaigns")
    p_dps.add_argument("--out", default=None,
                       help="panel JSON output (default ./debug_panel.json)")
    p_dps.set_defaults(func=cmd_debug_panel_score)

    for name, milestone in _STUBS.items():
        p_stub = sub.add_parser(name, help=f"(not implemented — milestone {milestone})")
        p_stub.set_defaults(func=_make_stub(name, milestone))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
