"""DeCART2D lattice deck generation + (parallel) runner (plan 12.1).

Template families live under two roots:

    <apr1400>/5.8_5.1/FA/IGD_{12,16,20}/{gd}_{n}_z{1,2}/dec_FA_*.inp
    <apr1400>/260624/FA/IGD_24/{gd}_{n}_z{1,2}/dec_FA_*.inp

Selecting a template by ``(gd_wt, n_gd, zoning_variant)`` fixes the Gd-pin count
and the edge-zoning arrangement (both encoded in the assembly pin map, which we
never touch).  Only three numeric MATERIAL edits differ per design: the ``UO2``
92235 (e1), the ``UO2_2`` 92235 (e2), and the ``UO2G`` ``6408`` Gd2O3 wt%.  The
DeCART product ``<CASEID>_0101.HGC`` is renamed to ``FA_<alias>.HGC`` (the ga80
5-char COMP-name convention), and the companion ``<CASEID>.out`` to
``FA_<alias>.out`` for the MASS(g) inventory parser.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .._proc import no_window_flags
from .spec import DesignRegistry, FuelDesign

DEFAULT_DECART_EXE = r"D:\DeCART_MASTER\BIN\decart2d1.1m5omp.exe"

#: (library subtree, IGD groups) to search for a ``{gd}_{n}_z{z}`` template dir.
_TEMPLATE_ROOTS = (
    ("5.8_5.1/FA", (12, 16, 20)),
    ("260624/FA", (24,)),
)


class LatticeError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# template resolution
# --------------------------------------------------------------------------- #
def _dir_name(design: FuelDesign) -> str:
    return f"{int(round(design.gd_wt))}_{design.n_gd}_{design.zoning_variant}"


def resolve_template(design: FuelDesign, apr1400_root: str | Path) -> Path:
    """Locate the ``dec_FA_*.inp`` template for a design's (gd_wt, n_gd, z)."""
    apr = Path(apr1400_root)
    want = _dir_name(design)
    for subtree, groups in _TEMPLATE_ROOTS:
        if design.n_gd not in groups:
            continue
        cand = apr / subtree / f"IGD_{design.n_gd}" / want
        if cand.is_dir():
            decks = sorted(cand.glob("dec_FA_*.inp"))
            if decks:
                return decks[0]
    # broad fallback: scan every IGD_* for the directory name.
    for subtree, _groups in _TEMPLATE_ROOTS:
        for igd in sorted((apr / subtree).glob("IGD_*")):
            cand = igd / want
            if cand.is_dir():
                decks = sorted(cand.glob("dec_FA_*.inp"))
                if decks:
                    return decks[0]
    raise LatticeError(
        f"no dec_FA template for design {design.type_id} (dir {want!r}) under {apr}"
    )


# --------------------------------------------------------------------------- #
# deck editing
# --------------------------------------------------------------------------- #
def _read_text_flex(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _caseid_of(text: str) -> str:
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0].upper() == "CASEID":
            return toks[1]
    raise LatticeError("template has no CASEID line")


def _sub_after(line: str, marker: str, value: str) -> str:
    """Replace the numeric token that follows ``marker`` on a line, keeping the
    exact surrounding whitespace."""
    pat = re.compile(rf"(\b{re.escape(marker)}\s+)([+-]?\d+(?:\.\d+)?)")
    new, n = pat.subn(rf"\g<1>{value}", line, count=1)
    if n == 0:
        raise LatticeError(f"marker {marker!r} not found on line: {line!r}")
    return new


def _fmt_enr(value: float) -> str:
    """Format an enrichment/wt%% compactly but always with a decimal point
    (Fortran list-directed reads are happiest with an explicit real)."""
    s = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s or "0.0"


def edit_dec_text(template_text: str, design: FuelDesign, new_caseid: str) -> str:
    """Return the DeCART deck text for ``design`` from a template.

    Edits: rename the CASEID token everywhere it appears; set ``UO2`` 92235=e1,
    ``UO2_2`` 92235=e2, and the ``UO2G`` ``6408``=gd_wt.  The pin map, Gd carrier
    enrichment (4.0), and all geometry stay byte-identical to the template.
    """
    old_caseid = _caseid_of(template_text)
    out_lines: list[str] = []
    in_gd = False
    for raw in template_text.splitlines():
        line = raw
        stripped = line.strip()
        if stripped.startswith("mixture UO2_2"):
            line = _sub_after(line, "92235", _fmt_enr(design.e2))
            in_gd = False
        elif stripped.startswith("mixture UO2G"):
            in_gd = True                        # 6408 sits on the next line
        elif stripped.startswith("mixture UO2"):
            line = _sub_after(line, "92235", _fmt_enr(design.e1))
            in_gd = False
        elif in_gd and "6408" in line.split():
            line = _sub_after(line, "6408", _fmt_enr(design.gd_wt))
            in_gd = False
        elif in_gd and stripped.startswith("mixture"):
            in_gd = False
        out_lines.append(line)

    text = "\n".join(out_lines)
    if template_text.endswith("\n"):
        text += "\n"
    # rename the CASEID token (whole word) everywhere: CASEID line, assembly,
    # rad_conf, and the standalone FA name line.
    text = re.sub(rf"(?<![\w]){re.escape(old_caseid)}(?![\w])", new_caseid, text)
    return text


# --------------------------------------------------------------------------- #
# GEOM editing (pin pitch + fuel-pin radii) — review sec. 3.3 / 4c.
#
# The MATERIAL editor above keeps ALL geometry byte-identical.  A pin-pitch /
# pin-radius optimization additionally needs to move the FIRST ``pitch`` token
# (pin pitch) and the ``cellgeo 1`` / ``cellgeo 2`` leading radii (r_pellet,
# r_clad_in, r_clad_out — the normal-fuel and IGD fuel pins).  It must NEVER touch
# the SECOND ``pitch`` token (assembly pitch 20.7772 == coredeck CoreParams.wide,
# the frozen envelope that keeps the MASTER core model valid) nor the guide-tube
# ``cellgeo 3-6``.  ``edit_dec_geom_text`` enforces exactly that with a hard guard.
# --------------------------------------------------------------------------- #
_NUM = r"[+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?"
_PITCH_T1_RE = re.compile(rf"(?im)^(\s*pitch\s+)({_NUM})(\s+)({_NUM})")
_CELLGEO_R3_RE = re.compile(
    rf"(?im)^(\s*cellgeo\s+[12]\s+)({_NUM})(\s+)({_NUM})(\s+)({_NUM})")
_CELLGEO_ANY_RE = re.compile(rf"(?im)^\s*cellgeo\s+(\d+)\b.*$")
_ASM_PITCH_TOL = 1.0e-6


def _fmt_geom(value: float) -> str:
    """Compact real with an explicit decimal point (Fortran list-directed read)."""
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s or "0.0"


def _geom_cellgeo_lines(text: str, ids: set[int]) -> list[str]:
    """The verbatim ``cellgeo`` lines whose id is in ``ids`` (frozen-check anchor)."""
    return [m.group(0) for m in _CELLGEO_ANY_RE.finditer(text)
            if int(m.group(1)) in ids]


def edit_dec_geom_text(
    template_text: str,
    *,
    pin_pitch: float | None = None,
    r_pellet: float | None = None,
    r_clad_in: float | None = None,
    r_clad_out: float | None = None,
) -> str:
    """Return deck text with the pin pitch and/or fuel-pin radii edited.

    Edits the first ``pitch`` token (pin pitch) and the three leading radii of the
    fuel pins ``cellgeo 1`` and ``cellgeo 2`` (kept identical across the two, as in
    every reference deck).  Any argument left ``None`` is not touched.  After
    editing it HARD-ASSERTS the invariants the MASTER core model depends on:

    * the second ``pitch`` token (assembly pitch) is byte-identical to the template;
    * the guide-tube ``cellgeo 3-6`` lines are byte-identical to the template.

    Raises :class:`LatticeError` if the deck has no ``pitch``/``cellgeo 1`` card or
    if an edit would violate either invariant.
    """
    orig_asm = None
    m = _PITCH_T1_RE.search(template_text)
    if m is None:
        raise LatticeError("deck has no editable 'pitch <pin> <asm>' card")
    orig_asm = float(m.group(4))
    if not _CELLGEO_R3_RE.search(template_text):
        raise LatticeError("deck has no 'cellgeo 1/2 <r1> <r2> <r3>' fuel-pin card")
    frozen_before = _geom_cellgeo_lines(template_text, {3, 4, 5, 6})

    text = template_text
    if pin_pitch is not None:
        text = _PITCH_T1_RE.sub(
            lambda mm: f"{mm.group(1)}{_fmt_geom(pin_pitch)}{mm.group(3)}{mm.group(4)}",
            text, count=1,
        )
    if r_pellet is not None or r_clad_in is not None or r_clad_out is not None:
        def _sub_radii(mm: re.Match) -> str:
            r1 = _fmt_geom(r_pellet) if r_pellet is not None else mm.group(2)
            r2 = _fmt_geom(r_clad_in) if r_clad_in is not None else mm.group(4)
            r3 = _fmt_geom(r_clad_out) if r_clad_out is not None else mm.group(6)
            return (f"{mm.group(1)}{r1}{mm.group(3)}{r2}{mm.group(5)}{r3}")
        text = _CELLGEO_R3_RE.sub(_sub_radii, text)

    # -- hard guard: the frozen envelope + guide tubes are untouched -----------
    m2 = _PITCH_T1_RE.search(text)
    if m2 is None or abs(float(m2.group(4)) - orig_asm) > _ASM_PITCH_TOL:
        raise LatticeError(
            "geometry edit changed the assembly-pitch token (frozen envelope); "
            "only the FIRST pitch token (pin pitch) may move"
        )
    frozen_after = _geom_cellgeo_lines(text, {3, 4, 5, 6})
    if frozen_after != frozen_before:
        raise LatticeError("geometry edit changed a guide-tube cellgeo 3-6 card (frozen)")
    return text


def write_dec_deck(design: FuelDesign, out_dir: str | Path, registry: DesignRegistry,
                   apr1400_root: str | Path) -> Path:
    """Write ``dec_FA_<alias>.inp`` for ``design`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alias = registry.alias(design)
    caseid = f"FA_{alias}"
    template = resolve_template(design, apr1400_root)
    text = edit_dec_text(_read_text_flex(template), design, caseid)
    deck_path = out / f"dec_FA_{alias}.inp"
    deck_path.write_text(text, encoding="utf-8")
    return deck_path


# --------------------------------------------------------------------------- #
# DeCART runner
# --------------------------------------------------------------------------- #
@dataclass
class DecartRun:
    """A launched DeCART2D run (detached) plus its bookkeeping."""

    design: FuelDesign
    alias: str
    work_dir: Path
    caseid: str
    fa_name: str
    process: subprocess.Popen | None = None
    started: float = 0.0
    wall_s: float | None = None
    returncode: int | None = None
    hgc_path: Path | None = None
    out_path: Path | None = None
    error: str | None = None

    @property
    def raw_hgc(self) -> Path:
        return self.work_dir / f"{self.caseid}_0101.HGC"

    def poll(self) -> bool:
        """True once the process has exited; records wall + returncode."""
        if self.process is None:
            return True
        rc = self.process.poll()
        if rc is None:
            return False
        if self.wall_s is None:
            self.wall_s = time.monotonic() - self.started
            self.returncode = rc
        return True


def _stage_deck(deck_path: Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(deck_path, work_dir / "decart.inp")


def launch_decart(deck_path: str | Path, work_dir: str | Path, design: FuelDesign,
                  alias: str, exe: str | Path = DEFAULT_DECART_EXE) -> DecartRun:
    """Copy the deck to ``decart.inp`` and launch DeCART2D detached."""
    deck_path = Path(deck_path)
    work_dir = Path(work_dir)
    caseid = _caseid_of(_read_text_flex(deck_path))
    run = DecartRun(design=design, alias=alias, work_dir=work_dir,
                    caseid=caseid, fa_name=f"FA_{alias}")
    if run.raw_hgc.exists():
        run.raw_hgc.unlink()                    # a stale product masks a failed run
    _stage_deck(deck_path, work_dir)
    log = open(work_dir / "decart.stdout", "wb")
    run.started = time.monotonic()
    run.process = subprocess.Popen(
        [str(exe)], cwd=str(work_dir), stdout=log, stderr=subprocess.STDOUT,
        **no_window_flags(),
    )
    return run


def harvest(run: DecartRun) -> DecartRun:
    """Rename the DeCART products to ``FA_<alias>.HGC`` / ``FA_<alias>.out``."""
    wd = run.work_dir
    raw = run.raw_hgc
    if not raw.is_file():
        run.error = f"no product {raw.name} (rc={run.returncode})"
        return run
    hgc = wd / f"{run.fa_name}.HGC"
    if hgc.exists():
        hgc.unlink()
    raw.rename(hgc)
    run.hgc_path = hgc

    # DeCART writes <caseid>.out; fall back to decart.out / any fresh *.out.
    out_src = None
    for cand in (wd / f"{run.caseid}.out", wd / "decart.out"):
        if cand.is_file():
            out_src = cand
            break
    if out_src is None:
        outs = [p for p in wd.glob("*.out") if p.name != f"{run.fa_name}.out"]
        out_src = outs[0] if outs else None
    if out_src is not None:
        out_dst = wd / f"{run.fa_name}.out"
        if out_dst != out_src:
            if out_dst.exists():
                out_dst.unlink()
            shutil.copyfile(out_src, out_dst)
        run.out_path = out_dst
    (wd / "decart.inp").unlink(missing_ok=True)
    return run


def run_decart(deck_path: str | Path, work_dir: str | Path, design: FuelDesign,
               alias: str, *, exe: str | Path = DEFAULT_DECART_EXE,
               detached: bool = False, timeout_s: float = 5400.0) -> DecartRun:
    """Run DeCART2D on one deck.

    ``detached=True`` returns immediately with a live :class:`DecartRun` (poll
    ``run.poll()`` then :func:`harvest`).  ``detached=False`` blocks up to
    ``timeout_s``, harvests, and returns the finished run.
    """
    run = launch_decart(deck_path, work_dir, design, alias, exe=exe)
    if detached:
        return run
    try:
        run.returncode = run.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        run.process.kill()
        run.process.wait()
        run.error = f"DeCART timed out after {timeout_s:g}s"
        run.wall_s = time.monotonic() - run.started
        return run
    run.wall_s = time.monotonic() - run.started
    return harvest(run)


def _hgc_looks_valid(hgc_path: Path) -> bool:
    """Cheap structural check that a staged ``FA_<alias>.HGC`` is a COMPLETE
    DeCART product (the idempotent-skip guard for :func:`run_batch`).

    A complete product is readable, non-trivially sized, and carries at least one
    ``%TITL`` state block with a ``CASE ::`` label and a ``%DIST`` pin-power map —
    exactly the structure the ``fuel_types`` HGC harvest depends on.  A truncated
    or empty product from a crashed run fails these and is re-run.
    """
    try:
        if not hgc_path.is_file() or hgc_path.stat().st_size < 256:
            return False
        text = _read_text_flex(hgc_path)
    except OSError:
        return False
    return ("%TITL" in text) and ("CASE ::" in text) and ("%DIST" in text)


def _completed_run(design: FuelDesign, alias: str, wd: Path) -> DecartRun | None:
    """Reuse a previously produced run: return a finished :class:`DecartRun` when
    ``wd`` already holds a valid ``FA_<alias>.HGC`` **and** its ``FA_<alias>.out``
    companion (both needed by the library build + fuel_types ingest); else None."""
    hgc = wd / f"FA_{alias}.HGC"
    out = wd / f"FA_{alias}.out"
    if not (_hgc_looks_valid(hgc) and out.is_file()):
        return None
    run = DecartRun(design=design, alias=alias, work_dir=wd,
                    caseid=f"FA_{alias}", fa_name=f"FA_{alias}")
    run.hgc_path = hgc
    run.out_path = out
    run.wall_s = 0.0
    run.returncode = 0
    return run


def run_batch(designs: list[FuelDesign], out_root: str | Path, registry: DesignRegistry,
              apr1400_root: str | Path, *, exe: str | Path = DEFAULT_DECART_EXE,
              max_parallel: int = 4, poll_s: float = 15.0,
              timeout_s: float = 5400.0) -> list[DecartRun]:
    """Generate decks and run DeCART2D concurrently (bounded by ``max_parallel``).

    Idempotent: a design whose ``FA_<alias>.HGC`` (+ ``.out``) already exists under
    ``out_root/<alias>`` and parses is reused as-is (no DeCART re-launch), so a
    pre-generated band or a crash-resumed batch does not recompute finished
    lattices.  Measures wall time per run.  Returns the finished
    :class:`DecartRun` list in the same order as ``designs``.
    """
    out_root = Path(out_root)
    pending = list(designs)
    active: list[DecartRun] = []
    done: dict[str, DecartRun] = {}

    while pending or active:
        while pending and len(active) < max_parallel:
            design = pending.pop(0)
            alias = registry.alias(design)
            wd = out_root / alias
            cached = _completed_run(design, alias, wd)
            if cached is not None:
                done[alias] = cached          # idempotent skip: reuse the product
                continue
            deck = write_dec_deck(design, wd, registry, apr1400_root)
            active.append(launch_decart(deck, wd, design, alias, exe=exe))

        time.sleep(poll_s)
        still: list[DecartRun] = []
        for run in active:
            timed_out = (time.monotonic() - run.started) > timeout_s
            if run.poll():
                harvest(run)
                done[run.alias] = run
            elif timed_out:
                run.process.kill()
                run.process.wait()
                run.wall_s = time.monotonic() - run.started
                run.error = f"DeCART timed out after {timeout_s:g}s"
                done[run.alias] = run
            else:
                still.append(run)
        active = still

    return [done[registry.alias(d)] for d in designs]


__all__ = [
    "DEFAULT_DECART_EXE",
    "DecartRun",
    "LatticeError",
    "edit_dec_geom_text",
    "edit_dec_text",
    "harvest",
    "launch_decart",
    "resolve_template",
    "run_batch",
    "run_decart",
    "write_dec_deck",
]
