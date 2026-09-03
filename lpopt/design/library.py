"""MASTER cross-section library build via the TotalBatcher4 chain (plan 12.1).

Ported from ``2_LP/MOCHA/library.py`` ``build_library``: stage ``MAS_REF`` (the
5 reflector COMP blocks) + every ``FA_<alias>.HGC`` + ``prolog41m4.exe`` +
``TotalBatcher4.exe`` into one directory and run TotalBatcher, which emits

    MAS_XSL   reflector blocks + one ``COMP FA_<alias>`` XSD block per FA
    MAS_HFF   one pin form-function block per FA

The build is verified to contain every requested set (``COMP FA_<alias>`` in
MAS_XSL, alias in MAS_HFF) and its COMP count is returned so the caller can probe
the MASTER ``ncomp`` ceiling (plan 12.1: split by enrichment band if exceeded).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .._proc import no_window_flags

_DECART_BIN = Path(r"D:\DeCART_MASTER\BIN")
#: The TotalBatcher/PROLOG binaries + a MAS_REF live as copies in every hgc dir;
#: 260624/hgc is locally hydrated and used as the default source.
_HGC_TOOLDIR_REL = "260624/hgc"


class LibraryBuildError(RuntimeError):
    pass


@dataclass
class LibraryBuild:
    """Result of a TotalBatcher library build."""

    xsl_path: Path
    hff_path: Path
    comp_count: int              # number of COMP (fuel) blocks in MAS_XSL
    refl_count: int              # number of REFL (reflector) blocks in MAS_XSL
    set_names: list[str]         # FA_<alias> set names present, in file order
    staging_dir: Path

    @property
    def ncomp(self) -> int:
        """MASTER ``ncomp`` = reflector COMPs (5) + fuel COMPs (deck GEN_DIM)."""
        return self.refl_count + self.comp_count


def default_tool_paths(apr1400_root: str | Path) -> dict[str, Path]:
    """Resolve MAS_REF / prolog / TotalBatcher from the workspace."""
    tdir = Path(apr1400_root) / _HGC_TOOLDIR_REL
    prolog = _DECART_BIN / "prolog41m4.exe"
    if not prolog.is_file():
        prolog = tdir / "prolog41m4.exe"
    return {
        "mas_ref": tdir / "MAS_REF",
        "prolog_exe": prolog,
        "totalbatcher_exe": tdir / "TotalBatcher4.exe",
    }


def _count_blocks(xsl_text: str) -> tuple[int, int, list[str]]:
    comp = 0
    refl = 0
    names: list[str] = []
    for line in xsl_text.splitlines():
        if line.startswith("COMP "):
            comp += 1
            toks = line.split()
            if len(toks) >= 2:
                names.append(toks[1])
        elif line.startswith("REFL "):
            refl += 1
    return comp, refl, names


def build_master_library(hgc_paths: list[str | Path], out_dir: str | Path,
                         *, mas_ref: str | Path, prolog_exe: str | Path,
                         totalbatcher_exe: str | Path, library_id: str = "paramA",
                         timeout_s: float = 3600.0,
                         snapshot_dir: str | Path | None = None) -> LibraryBuild:
    """Build ``MAS_XSL`` / ``MAS_HFF`` from ``hgc_paths`` into ``out_dir``.

    ``hgc_paths`` must be named ``FA_<alias>.HGC`` (the stem becomes the MASTER
    set name).  Existing products are backed up (``.bak``) rather than deleted so
    a production dir passed as ``out_dir`` survives a failed rebuild.  Raises
    :class:`LibraryBuildError` if a requested set is missing from either product.

    ``snapshot_dir`` (task #12) gates the rebuild: it is :func:`require_snapshot`
    -checked against the package root (``out_dir``'s parent, since ``out_dir`` is
    ``<pkg>/lib``) BEFORE the single ``.bak`` generation is destroyed, so a
    rebuild whose rollback is missing or stale raises :class:`SnapshotError` with
    the package still intact.  Omitted (the default) nothing changes.
    """
    staging = Path(out_dir)
    if snapshot_dir is not None:
        require_snapshot(staging.parent, snapshot_dir)
    staging.mkdir(parents=True, exist_ok=True)

    for fn in ("MAS_XSL", "MAS_HFF"):
        p = staging / fn
        if p.exists():
            bak = staging / (fn + ".bak")
            if bak.exists():
                bak.unlink()
            p.rename(bak)

    expected = {Path(h).name for h in hgc_paths}
    stale = [p.name for p in staging.glob("*.HGC") if p.name not in expected]
    stale += [p.name for p in staging.glob("*.hgc") if p.name not in expected]
    if stale:
        raise LibraryBuildError(
            f"staging dir {staging} has HGC files not in the request: {sorted(stale)}"
        )

    for src in [mas_ref, prolog_exe, totalbatcher_exe, *hgc_paths]:
        src = Path(src)
        if not src.is_file():
            raise LibraryBuildError(f"missing input: {src}")
        dst = staging / src.name
        if dst.resolve() != src.resolve():
            shutil.copyfile(src, dst)

    # TotalBatcher shells out to 'prolog41m4.exe' by bare name; prepend the
    # staging dir to PATH so the cwd-independent lookup finds it.
    env = dict(os.environ)
    env["PATH"] = str(staging.resolve()) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [str(staging / Path(totalbatcher_exe).name)],
        cwd=str(staging), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, timeout=timeout_s, **no_window_flags(),
    )
    xsl, hff = staging / "MAS_XSL", staging / "MAS_HFF"
    if not xsl.is_file() or not hff.is_file():
        tail = proc.stdout.decode(errors="replace")[-3000:] if proc.stdout else ""
        raise LibraryBuildError(
            f"TotalBatcher produced no MAS_XSL/MAS_HFF (rc={proc.returncode})\n{tail}"
        )

    xsl_text = xsl.read_text(errors="replace")
    hff_text = hff.read_text(errors="replace")
    for h in hgc_paths:
        name = Path(h).stem
        if f"COMP {name}" not in xsl_text:
            raise LibraryBuildError(f"set {name} missing from MAS_XSL")
        if name not in hff_text:
            raise LibraryBuildError(f"set {name} missing from MAS_HFF")

    comp, refl, names = _count_blocks(xsl_text)
    if comp != len(expected):
        raise LibraryBuildError(
            f"MAS_XSL COMP count {comp} != {len(expected)} requested FA sets"
        )
    return LibraryBuild(xsl_path=xsl, hff_path=hff, comp_count=comp, refl_count=refl,
                        set_names=names, staging_dir=staging)


# --------------------------------------------------------------------------- #
# pre-rebuild snapshot (task #12)
# --------------------------------------------------------------------------- #
#: What a rebuild invalidates and what a rollback therefore needs: the products
#: (``lib/``), everything keyed to them (``bases/`` restarts, ``cores/``
#: templates) and the two identity files.  Precedent: ``lib.snap_20260811``.
SNAPSHOT_MEMBERS: tuple[str, ...] = (
    "lib", "bases", "cores", "registry.json", "designs.json",
)

#: Name of the sha-256 manifest written beside the archive.
MANIFEST_NAME = "sha256_manifest.json"


class SnapshotError(RuntimeError):
    """The pre-rebuild snapshot is missing, incomplete, or does not match."""


@dataclass
class PackageSnapshot:
    """A taken snapshot: its directory, archive, and file hashes."""

    tag: str
    snapshot_dir: Path
    archive_path: Path
    manifest_path: Path
    hashes: dict[str, str] = field(default_factory=dict)

    @property
    def n_files(self) -> int:
        return len(self.hashes)


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming sha-256 of a file (packages hold multi-GB restarts)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def package_hashes(pkg_dir: str | Path,
                   members: Sequence[str] = SNAPSHOT_MEMBERS) -> dict[str, str]:
    """``{posix relative path: sha256}`` over ``members`` of the package.

    A member that does not exist contributes nothing (a package with no
    ``cores/`` yet is snapshot-able); the manifest records what WAS there.
    """
    pkg = Path(pkg_dir)
    out: dict[str, str] = {}
    for member in members:
        src = pkg / member
        if src.is_file():
            out[member] = sha256_file(src)
        elif src.is_dir():
            for p in sorted(src.rglob("*")):
                if p.is_file():
                    out[p.relative_to(pkg).as_posix()] = sha256_file(p)
    return out


def _archive_member_names(archive: str | Path) -> set[str]:
    """Posix member paths inside a snapshot archive (tar* or zip)."""
    import tarfile
    import zipfile

    path = Path(archive)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    else:
        with tarfile.open(path) as tf:
            names = tf.getnames()
    return {n.lstrip("./").rstrip("/") for n in names}


def snapshot_package(pkg_dir: str | Path, dest_root: str | Path, *,
                     tag: str | None = None,
                     members: Sequence[str] = SNAPSHOT_MEMBERS,
                     archive_format: str = "gztar") -> PackageSnapshot:
    """Archive the rebuild-critical package members + a sha-256 manifest.

    Written to ``<dest_root>/<tag>/`` (``tag`` defaults to a UTC timestamp), the
    archive next to :data:`MANIFEST_NAME`.  This is the ONLY rollback that
    survives a second rebuild: :func:`build_master_library` keeps exactly ONE
    ``.bak`` generation and the next rebuild unlinks it, so a rebuild that is
    not snapshotted first is irreversible after the one that follows it.

    Refuses to overwrite an existing ``<dest_root>/<tag>/`` — a tag is an
    immutable rollback point.
    """
    pkg = Path(pkg_dir)
    if not pkg.is_dir():
        raise SnapshotError(f"package {pkg} does not exist")
    tag = tag or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    snap_dir = Path(dest_root) / tag
    if snap_dir.exists() and any(snap_dir.iterdir()):
        raise SnapshotError(
            f"snapshot {snap_dir} already exists — a tag is an immutable "
            f"rollback point; pick a new tag")
    snap_dir.mkdir(parents=True, exist_ok=True)

    hashes = package_hashes(pkg, members)
    if not hashes:
        raise SnapshotError(f"package {pkg} holds none of {list(members)}")

    # Stage only the requested members, then archive the staging tree, so the
    # archive can never pick up bootstrap_work/ or a stale .snap_* sibling.
    stage = snap_dir / "_stage"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        for member in members:
            src = pkg / member
            if src.is_file():
                shutil.copy2(src, stage / src.name)
            elif src.is_dir():
                shutil.copytree(src, stage / member, dirs_exist_ok=True)
        archive = Path(shutil.make_archive(str(snap_dir / "package"),
                                           archive_format, root_dir=str(stage)))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    # The hashes were taken BEFORE staging, so "the manifest verifies" does not
    # by itself prove the archive HOLDS those files: a member that failed to
    # stage (or a file that moved mid-snapshot) would yield a snapshot that
    # verifies clean and restores wrong — and this is the only rollback that
    # survives a second rebuild.  Read the archive back and require every
    # recorded path to be in it.
    archive_members = _archive_member_names(archive)
    missing = sorted(rel for rel in hashes if rel not in archive_members)
    if missing:
        raise SnapshotError(
            f"snapshot archive {archive.name} is incomplete: "
            f"{len(missing)} recorded file(s) absent, e.g. {missing[:5]}")

    manifest = snap_dir / MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "tag": tag,
                "package_root": str(pkg.resolve()),
                "members": list(members),
                "archive": archive.name,
                "archive_sha256": sha256_file(archive),
                "archive_members": sorted(archive_members),
                "files": dict(sorted(hashes.items())),
            },
            indent=2, sort_keys=False,
        ) + "\n",
        encoding="utf-8",
    )
    return PackageSnapshot(tag=tag, snapshot_dir=snap_dir, archive_path=archive,
                           manifest_path=manifest, hashes=hashes)


def verify_snapshot(pkg_dir: str | Path, snapshot_dir: str | Path) -> list[str]:
    """Differences between the live package and a snapshot manifest.

    Returns a list of human-readable problems — empty means the snapshot is
    present, its archive is intact, and every recorded file still hashes the
    same on disk.
    """
    snap = Path(snapshot_dir)
    manifest = snap / MANIFEST_NAME
    if not manifest.is_file():
        return [f"no {MANIFEST_NAME} in {snap}"]
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    problems: list[str] = []

    archive = snap / str(doc.get("archive", ""))
    if not archive.is_file():
        problems.append(f"snapshot archive {archive} is missing")
    elif doc.get("archive_sha256") and sha256_file(archive) != doc["archive_sha256"]:
        problems.append(f"snapshot archive {archive.name} sha256 mismatch")
    else:
        # The archive is the rollback; prove it CONTAINS every recorded file
        # rather than only that it hashes to what was recorded.
        members = _archive_member_names(archive)
        absent = sorted(rel for rel in doc.get("files", {}) if rel not in members)
        if absent:
            problems.append(
                f"snapshot archive {archive.name} is missing {len(absent)} "
                f"recorded file(s), e.g. {absent[:5]}")

    pkg = Path(pkg_dir)
    for rel, digest in sorted(doc.get("files", {}).items()):
        live = pkg / rel
        if not live.is_file():
            problems.append(f"{rel}: in snapshot, missing from the package")
        elif sha256_file(live) != digest:
            problems.append(f"{rel}: sha256 differs from the snapshot")
    return problems


def require_snapshot(pkg_dir: str | Path, snapshot_dir: str | Path) -> dict:
    """Gate a rebuild on a matching snapshot; raise :class:`SnapshotError`.

    Call this immediately before :func:`build_master_library` on a PRODUCTION
    package: it proves the rollback exists and still describes the package about
    to be overwritten.  Returns the manifest document on success.
    """
    problems = verify_snapshot(pkg_dir, snapshot_dir)
    if problems:
        raise SnapshotError(
            "pre-rebuild snapshot does not cover this package:\n  - "
            + "\n  - ".join(problems))
    return json.loads(
        (Path(snapshot_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# library / deck gates (task #15, prereg sec. 4.3 + 5)
# --------------------------------------------------------------------------- #
#: G-H3 size arithmetic, EXACT (no tolerance).  Verified on six library sizes
#: (N = 11/12/16/33/37/80) with zero error — prereg sec. 4.3.
#: The intercept is the 2,008-byte ``MAS_REF`` prolog block **plus its CRLF**.
MAS_XSL_INTERCEPT_BYTES = 2_010
#: Per-set ``MAS_XSL`` cost of a gadolinia-bearing set...
MAS_XSL_BYTES_PER_GD_SET = 385_849
#: ...and of a set with ``n_gd == 0`` (the seq_canary V01/V02 precedent).
MAS_XSL_BYTES_PER_NOGD_SET = 377_461
#: ``MAS_HFF`` is one pin form-function block per set, poison-independent.
MAS_HFF_BYTES_PER_SET = 404_857

#: G-H3b: the ``BURN VAR DMOD ADF DUM`` header every fuel COMP block carries.
COMP_HEADER = (62, 17, 6, 0, 0)

#: A nuclide/material label line inside a COMP block: a 4-character name in
#: column 1 followed by ``*`` (``U235*``, ``BP01*``, ``SB10*``, ``MACX*``,
#: ``CRD1*``, ``FISP*``, ``RESI*``, ``H2O *``).
_NUCLIDE_RE = re.compile(r"^(?P<name>[A-Z][A-Z0-9 ]{3})\*")
_COMP_HEADER_LABEL_RE = re.compile(r"^\s*BURN\s+VAR\s+DMOD\s+ADF\s+DUM\s*$")


class LibraryGateError(AssertionError):
    """A library/deck gate of prereg sec. 4.3 / 5 failed."""


def expected_library_sizes(n_gd_sets: int, n_nogd_sets: int = 0) -> tuple[int, int]:
    """G-H3 expected ``(MAS_XSL, MAS_HFF)`` byte sizes for a roster.

    ``N = 39`` (37 + the slice-Z pair, all ``n_gd > 0``) ->
    ``(15_050_121, 15_789_423)``.
    """
    if n_gd_sets < 0 or n_nogd_sets < 0:
        raise ValueError("set counts must be non-negative")
    n = n_gd_sets + n_nogd_sets
    if n == 0:
        raise ValueError("a library has at least one FA set")
    xsl = (MAS_XSL_INTERCEPT_BYTES
           + MAS_XSL_BYTES_PER_GD_SET * n_gd_sets
           + MAS_XSL_BYTES_PER_NOGD_SET * n_nogd_sets)
    return xsl, MAS_HFF_BYTES_PER_SET * n


def gate_library_sizes(xsl_bytes: int, hff_bytes: int, n_gd_sets: int,
                       n_nogd_sets: int = 0) -> None:
    """**G-H3** — the two size equalities, with NO tolerance."""
    exp_xsl, exp_hff = expected_library_sizes(n_gd_sets, n_nogd_sets)
    problems = []
    if int(xsl_bytes) != exp_xsl:
        problems.append(f"MAS_XSL {int(xsl_bytes)} B != expected {exp_xsl} B "
                        f"(delta {int(xsl_bytes) - exp_xsl:+d})")
    if int(hff_bytes) != exp_hff:
        problems.append(f"MAS_HFF {int(hff_bytes)} B != expected {exp_hff} B "
                        f"(delta {int(hff_bytes) - exp_hff:+d})")
    if problems:
        raise LibraryGateError(
            f"G-H3 (N={n_gd_sets + n_nogd_sets}, {n_nogd_sets} with n_gd=0): "
            + "; ".join(problems))


def comp_blocks(xsl_text: str) -> dict[str, list[str]]:
    """``{set name: block lines}`` for every ``COMP <name>`` block in MAS_XSL."""
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in xsl_text.splitlines():
        if line.startswith("COMP ") or line.startswith("REFL "):
            toks = line.split()
            current = [] if (line.startswith("COMP ") and len(toks) >= 2) else None
            if current is not None:
                blocks[toks[1]] = current
        if current is not None:
            current.append(line)
    return blocks


def comp_nuclide_roster(block_lines: Iterable[str]) -> tuple[str, ...]:
    """The ordered nuclide/material roster of one COMP block.

    Counting the entries is not enough — the defect this gate exists for is a
    roster that silently drops a field a burnable-absorber set feeds
    (``BP01*``/``SB10*``/``MACX*``/``CRD1*``), which a count cannot see.
    """
    names: list[str] = []
    for line in block_lines:
        m = _NUCLIDE_RE.match(line)
        if m:
            names.append(m.group("name").strip())
    return tuple(names)


def comp_header_counts(block_lines: Iterable[str]) -> tuple[int, ...] | None:
    """``(BURN, VAR, DMOD, ADF, DUM)`` of one COMP block (``None`` if absent)."""
    lines = list(block_lines)
    for i, line in enumerate(lines):
        if _COMP_HEADER_LABEL_RE.match(line.split("#", 1)[0]):
            for data in lines[i + 1:]:
                toks = data.split()
                # PROLOG writes a ruler line between the labels and the counts.
                if not toks or set(data.strip()) <= {"-"}:
                    continue
                if all(t.lstrip("-").isdigit() for t in toks[:5]) and len(toks) >= 5:
                    return tuple(int(t) for t in toks[:5])
                break
            return None
    return None


def gate_comp_rosters(xsl_text: str, new_sets: Sequence[str], *,
                      expected_header: tuple[int, ...] = COMP_HEADER) -> None:
    """**G-H3b** — every new COMP block matches the incumbent blocks exactly.

    The nuclide roster of each set in ``new_sets`` must equal the roster shared
    by the pre-existing sets, and every block's header must be
    ``expected_header``.
    """
    blocks = comp_blocks(xsl_text)
    missing = [s for s in new_sets if s not in blocks]
    if missing:
        raise LibraryGateError(f"G-H3b: new set(s) absent from MAS_XSL: {missing}")
    incumbent = [n for n in blocks if n not in set(new_sets)]
    if not incumbent:
        raise LibraryGateError(
            "G-H3b needs at least one incumbent COMP block to compare against")

    rosters = {n: comp_nuclide_roster(blocks[n]) for n in blocks}
    reference = rosters[incumbent[0]]
    for name in incumbent[1:]:
        if rosters[name] != reference:
            raise LibraryGateError(
                f"G-H3b: incumbent sets disagree ({incumbent[0]} vs {name}) — "
                f"the library was already inconsistent before this build")
    for name in new_sets:
        if rosters[name] != reference:
            extra = sorted(set(rosters[name]) - set(reference))
            gone = sorted(set(reference) - set(rosters[name]))
            raise LibraryGateError(
                f"G-H3b: COMP {name} roster differs from the incumbent "
                f"({len(rosters[name])} vs {len(reference)} entries; "
                f"missing {gone}, unexpected {extra})")
    for name in blocks:
        header = comp_header_counts(blocks[name])
        if header != tuple(expected_header):
            raise LibraryGateError(
                f"G-H3b: COMP {name} header {header} != {tuple(expected_header)}")


def gate_comp_order(before: Sequence[str], after: Sequence[str],
                    new_sets: Sequence[str]) -> None:
    """**G-H3c** — the incumbent COMP order is unchanged, new sets append.

    R21: ``_ALIAS_LETTERS`` runs ``P..Z`` then ``A..O``, so a first ``A*`` alias
    could sort ahead of the incumbents.  TotalBatcher's internal ordering rule is
    not observable, so this checks the RESULT rather than asserting a mechanism.
    """
    before, after, new_sets = list(before), list(after), list(new_sets)
    if after[:len(before)] != before:
        for i, (b, a) in enumerate(zip(before, after)):
            if b != a:
                raise LibraryGateError(
                    f"G-H3c: COMP order changed at index {i}: {b!r} -> {a!r} "
                    f"(every MAS_RST.* is keyed to the old order)")
        raise LibraryGateError(
            f"G-H3c: rebuilt library has {len(after)} sets, fewer than the "
            f"{len(before)} it must preserve")
    appended = after[len(before):]
    if appended != new_sets:
        raise LibraryGateError(
            f"G-H3c: appended sets {appended} != requested {new_sets}")


def gate_cycle1_deck(deck: str, aliases: Sequence[str], *,
                     expected_dims: tuple[int, int],
                     xsl_text: str | None = None,
                     hff_text: str | None = None) -> None:
    """**G-H5a** — the cy1 fresh-core deck against the rebuilt library.

    ``validate_reload_deck`` MUST NOT be used here: it refuses any deck carrying
    ``%LPD_BCH``, which is exactly what a fresh-core cy1 deck is made of.  This
    gate instead checks ``%GEN_DIM``, that the ``%LPD_BCH`` batch roster names
    only declared aliases, and that the ``%LPD_C&X`` / ``%LPD_HFF`` set names
    exist in the products (when their text is supplied).
    """
    problems: list[str] = []
    dims = _deck_dims(deck)
    if dims is None:
        problems.append("no %GEN_DIM nbatch/ncomp line")
    elif dims != tuple(expected_dims):
        problems.append(f"%GEN_DIM {dims} != library {tuple(expected_dims)}")

    bch = _deck_map_tokens(deck, "LPD_BCH")
    if not bch:
        problems.append("no %LPD_BCH fresh-core batch map (not a cy1 deck)")
    known = set(aliases)
    # ``o`` = no cell, ``R*`` = reflector batch (coredeck.is_fuel); everything
    # else in the fresh-core map must be a declared library alias.
    unknown = sorted({b for b in bch
                      if b != "o" and not b.startswith("R") and b not in known})
    if unknown:
        problems.append(f"%LPD_BCH names batch id(s) outside the roster: {unknown}")

    for card, text, product in (("LPD_C&X", xsl_text, "MAS_XSL"),
                                ("LPD_HFF", hff_text, "MAS_HFF")):
        names = _deck_set_names(deck, card)
        if not names:
            problems.append(f"no %{card} set names")
        elif text is not None:
            absent = sorted(n for n in names if n not in text)
            if absent:
                problems.append(f"%{card} names not in {product}: {absent}")
    if problems:
        raise LibraryGateError("G-H5a (cy1 deck): " + "; ".join(problems))


def gate_reload_deck(deck: str, restart_basename: str, *,
                     expected_dims: tuple[int, int],
                     allowed_batch_ids: Sequence[str] | None = None) -> None:
    """**G-H5b** — a cy >= 2 deck through ``validate_reload_deck`` (its purpose)."""
    from ..search.assets import DeckValidationError, validate_reload_deck

    try:
        validate_reload_deck(deck, restart_basename,
                             expected_dims=tuple(expected_dims),
                             allowed_batch_ids=allowed_batch_ids)
    except DeckValidationError as exc:
        raise LibraryGateError(f"G-H5b (reload deck): {exc}") from exc


def gate_convergence(*, converged: bool, n_cycles: int,
                     converged_at_cap: bool = False,
                     max_cycles: int = 16, consecutive: int = 2) -> None:
    """**G-H5c** — the bootstrap chain settled INSIDE the cap.

    "Equilibrium" is not ">= 2 cycles": it is ``make_band_restart``'s five-FOM
    comparison stable ``consecutive`` times within ``max_cycles`` (T6_T4 needed
    11).  A chain that only ran out of cycles is a FAILED bootstrap.

    "Settled INSIDE the cap" is DERIVED from ``n_cycles`` rather than trusted
    from the caller: a chain whose last cycle is the cap is the case prereg line
    339 excludes, and it used to pass whenever the caller forgot the flag.
    ``converged_at_cap=True`` still forces the finding for a caller that knows
    more than the cycle count.
    """
    problems = []
    if not converged:
        problems.append(
            f"chain did not settle in {n_cycles} of at most {max_cycles} cycles")
    at_cap = bool(converged_at_cap) or (converged and n_cycles >= max_cycles)
    if at_cap:
        problems.append("converged only AT the cycle cap (not a settled chain)")
    if converged and n_cycles < consecutive:
        problems.append(
            f"{n_cycles} cycles cannot show {consecutive} consecutive stable "
            f"comparisons")
    if n_cycles > max_cycles:
        problems.append(f"{n_cycles} cycles exceeds max_cycles={max_cycles}")
    if problems:
        raise LibraryGateError("G-H5c (convergence): " + "; ".join(problems))


# -- small deck readers used by the gates (no import of the search harness) -- #
def _deck_data_lines(deck: str, card: str) -> list[str]:
    out: list[str] = []
    lines = deck.splitlines()
    target = "%" + card.upper()
    for i, line in enumerate(lines):
        if line.split("#", 1)[0].strip().upper() != target:
            continue
        for data in lines[i + 1:]:
            if data.lstrip().startswith("%"):
                break
            body = data.split("#", 1)[0].rstrip()
            if body.strip():
                out.append(body)
    return out


def _deck_dims(deck: str) -> tuple[int, int] | None:
    for body in _deck_data_lines(deck, "GEN_DIM"):
        toks = body.split()
        if len(toks) >= 5 and all(t.lstrip("-").isdigit() for t in toks[:5]):
            return int(toks[3]), int(toks[4])
        return None
    return None


def _deck_card_ids(deck: str, card: str) -> tuple[str, ...]:
    ids = []
    for body in _deck_data_lines(deck, card):
        toks = body.split()
        if toks and toks[0] != "/":
            ids.append(toks[0])
    return tuple(ids)


def _deck_map_tokens(deck: str, card: str) -> tuple[str, ...]:
    """Every token of a 2-D card body (``%LPD_BCH`` is a map, not a list)."""
    toks: list[str] = []
    for body in _deck_data_lines(deck, card):
        toks.extend(t for t in body.split() if t != "/")
    return tuple(toks)


def _deck_set_names(deck: str, card: str) -> tuple[str, ...]:
    names = []
    for body in _deck_data_lines(deck, card):
        toks = body.split()
        if len(toks) >= 2:
            names.append(toks[1])
    return tuple(names)


# --------------------------------------------------------------------------- #
# CLI hook (task #12): python -m lpopt.design.library snapshot|verify
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """``snapshot``/``verify`` a package before a TotalBatcher rebuild."""
    import argparse

    ap = argparse.ArgumentParser(prog="lpopt.design.library",
                                 description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot", help="archive lib/bases/cores + manifest")
    snap.add_argument("package", help="package root (data/design/package)")
    snap.add_argument("dest", help="archive root (e.g. E:\\lpopt_archive)")
    snap.add_argument("--tag", default=None, help="snapshot tag (default: UTC stamp)")
    snap.add_argument("--format", default="gztar",
                      choices=("gztar", "zip", "tar", "bztar", "xztar"))

    ver = sub.add_parser("verify", help="gate a rebuild on a matching snapshot")
    ver.add_argument("package")
    ver.add_argument("snapshot", help="the <dest>/<tag> directory")

    reg = sub.add_parser(
        "regen", help="regenerate cores/ templates against the assembled roster")
    reg.add_argument("package", help="package root (data/design/package)")
    reg.add_argument("--dry-run", action="store_true",
                     help="list the regeneration and write/delete nothing")
    reg.add_argument("--synth-root", default=None,
                     help="cached synth deck root to purge (data/design/synth_decks)")
    reg.add_argument("--no-purge-synth", action="store_true",
                     help="skip the cached-deck purge deliberately")
    reg.add_argument("--accept-stale-bases", action="store_true",
                     help="proceed although bases/ restarts must be re-bootstrapped")

    args = ap.parse_args(argv)
    if args.cmd == "regen":
        from .bootstrap import library_aliases
        from .package import StaleBasesError, regenerate_core_templates

        aliases = library_aliases(args.package)
        try:
            report = regenerate_core_templates(
                args.package, aliases, dry_run=args.dry_run,
                synth_root=args.synth_root,
                purge_synth=not args.no_purge_synth,
                accept_stale_bases=args.accept_stale_bases)
        except StaleBasesError as exc:
            print(json.dumps(exc.report.as_dict(), indent=2))
            print(f"REFUSED: {exc}")
            return 1
        print(json.dumps(report.as_dict(), indent=2))
        return 0
    if args.cmd == "snapshot":
        result = snapshot_package(args.package, args.dest, tag=args.tag,
                                  archive_format=args.format)
        print(json.dumps({"tag": result.tag, "archive": str(result.archive_path),
                          "manifest": str(result.manifest_path),
                          "n_files": result.n_files}, indent=2))
        return 0
    problems = verify_snapshot(args.package, args.snapshot)
    for p in problems:
        print(f"MISMATCH: {p}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


__all__ = [
    "COMP_HEADER",
    "MANIFEST_NAME",
    "MAS_HFF_BYTES_PER_SET",
    "MAS_XSL_BYTES_PER_GD_SET",
    "MAS_XSL_BYTES_PER_NOGD_SET",
    "MAS_XSL_INTERCEPT_BYTES",
    "SNAPSHOT_MEMBERS",
    "LibraryBuild",
    "LibraryBuildError",
    "LibraryGateError",
    "PackageSnapshot",
    "SnapshotError",
    "build_master_library",
    "comp_blocks",
    "comp_header_counts",
    "comp_nuclide_roster",
    "default_tool_paths",
    "expected_library_sizes",
    "gate_comp_order",
    "gate_comp_rosters",
    "gate_convergence",
    "gate_cycle1_deck",
    "gate_library_sizes",
    "gate_reload_deck",
    "main",
    "package_hashes",
    "require_snapshot",
    "sha256_file",
    "snapshot_package",
    "verify_snapshot",
]


if __name__ == "__main__":       # pragma: no cover - CLI entry
    raise SystemExit(main())
