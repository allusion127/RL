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

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
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
                         timeout_s: float = 3600.0) -> LibraryBuild:
    """Build ``MAS_XSL`` / ``MAS_HFF`` from ``hgc_paths`` into ``out_dir``.

    ``hgc_paths`` must be named ``FA_<alias>.HGC`` (the stem becomes the MASTER
    set name).  Existing products are backed up (``.bak``) rather than deleted so
    a production dir passed as ``out_dir`` survives a failed rebuild.  Raises
    :class:`LibraryBuildError` if a requested set is missing from either product.
    """
    staging = Path(out_dir)
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


__all__ = [
    "LibraryBuild",
    "LibraryBuildError",
    "build_master_library",
    "default_tool_paths",
]
