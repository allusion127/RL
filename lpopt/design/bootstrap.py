"""cy1 bootstrap + equilibrium chain for a paramA package (plan 12.1).

``make_band_restart`` seeds a new (pair, feed) cell:

  1. synthesize + run the cy1 fresh-core deck (no restart) -> first MAS_RST,
  2. drive the vendor ``EquilibriumRunner`` (reload template + a valid pattern)
     until the five-FOM comparison settles,
  3. save the converged chain's final ``MAS_RST.*`` to ``bases/<folder>/``.

The final restart is the band seed the produce/verify harness reuses (plan 5.2:
"밴드당 1개면 충분").  ``enable_pin_burnup`` turns on MAS_PPI near convergence so
``max_pin_burnup`` is parsed (plan 12.3, load-bearing for the 6.6 w/o pair).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .._proc import no_window_flags
from ..search.genome import (
    MAX_FRESH_TYPES,
    case_batches,
    fresh_units_from_feed,
    random_genome,
)
from ..vendor.masterrl.dataset import CaseData
from ..vendor.masterrl.domain import CaseKey
from ..vendor.masterrl.equilibrium import EquilibriumTolerances
from ..vendor.masterrl.master import MasterRunner
from ..search.verify import PurgingEquilibriumRunner
from .coredeck import DEFAULT_CORE, CoreParams, build_cycle1_deck, build_reload_deck

DEFAULT_MASTER_EXE = r"D:\DeCART_MASTER\BIN\master4.0m4_r1.exe"


class BootstrapError(RuntimeError):
    pass


@dataclass
class BootstrapResult:
    pair: str
    feed: int
    folder: str
    converged: bool
    converged_at_cap: bool
    n_cycles: int                    # reload cycles chained (excl. cy1)
    cycles_needed: int               # total MASTER runs incl. cy1
    restart_path: Path | None
    tolerance_margin: float | None
    wall_s: float
    # FOM (from the last cycle)
    cyclen: float | None = None
    f_r: float | None = None
    f_q: float | None = None
    cbc_max: float | None = None
    ao_abs: float | None = None
    max_assembly_burnup: float | None = None
    max_pin_burnup: float | None = None
    discharge_burnup: float | None = None
    error: str | None = None

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in (
            "pair", "feed", "folder", "converged", "converged_at_cap", "n_cycles",
            "cycles_needed", "cyclen", "f_r", "f_q", "cbc_max", "ao_abs",
            "max_assembly_burnup", "max_pin_burnup", "discharge_burnup",
            "tolerance_margin", "wall_s", "error")}


# --------------------------------------------------------------------------- #
# cy1 (no restart) — driven directly, since MasterRunner always stages a restart
# --------------------------------------------------------------------------- #
def run_cycle1(cy1_deck: str, xsl: Path, hff: Path, exe: str | Path, work_dir: Path,
               *, timeout_s: float = 3600.0) -> Path:
    """Run the fresh-core cy1 deck; return the produced ``MAS_RST.*`` path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "MAS_INP").write_text(cy1_deck, encoding="utf-8")
    for src, name in ((xsl, "MAS_XSL"), (hff, "MAS_HFF")):
        shutil.copyfile(src, work_dir / name)
    before = {p.name for p in work_dir.glob("MAS_RST.*")}
    log = work_dir / "MASTER.stdout"
    with open(log, "wb") as fh:
        proc = subprocess.run([str(exe)], cwd=str(work_dir), stdout=fh,
                              stderr=subprocess.STDOUT, timeout=timeout_s,
                              **no_window_flags())
    rst = [p for p in sorted(work_dir.glob("MAS_RST.*")) if p.name not in before]
    if not rst:
        tail = log.read_bytes()[-2000:].decode(errors="replace")
        raise BootstrapError(
            f"cy1 produced no MAS_RST.* (rc={proc.returncode})\n{tail}")
    if len(rst) != 1:
        raise BootstrapError(f"cy1 produced {len(rst)} MAS_RST.* files: {rst}")
    return rst[0]


# --------------------------------------------------------------------------- #
# discharge burnup estimate (FOM has no discharge_burnup field; derive it)
# --------------------------------------------------------------------------- #
def estimate_discharge_burnup(cyclen: float, feed: int, *, power_mw: float,
                              hm_mtu: float) -> float:
    """Energy-balance equilibrium discharge burnup [MWd/kgHM].

    per-cycle core-average increment = P_th * cyclen / HM; a batch resides
    ``241/feed`` cycles, so discharge ~ increment * residence.  Reported as an
    estimate — the vendor FOM carries no native discharge_burnup (plan 12.4
    promotes it to a first-class MASTER target).
    """
    if hm_mtu <= 0:
        return float("nan")
    per_cycle_mwd_per_mtu = power_mw * cyclen / hm_mtu
    residence = 241.0 / feed
    return per_cycle_mwd_per_mtu * residence / 1000.0     # MWd/MTU -> MWd/kgHM


def library_aliases(pkg_dir: str | Path) -> list[str]:
    """Read the FA_<alias> set names from ``lib/MAS_XSL`` (COMP order)."""
    xsl = Path(pkg_dir) / "lib" / "MAS_XSL"
    names = []
    for line in xsl.read_text(errors="replace").splitlines():
        if line.startswith("COMP FA_"):
            names.append(line.split()[1][len("FA_"):])
    if not names:
        raise BootstrapError(f"no COMP FA_* sets found in {xsl}")
    return names


# --------------------------------------------------------------------------- #
# the band-seed bootstrap
# --------------------------------------------------------------------------- #
def make_band_restart(pkg_dir: str | Path, pair: str, feed: int, rng,
                      *, aliases: list[str] | None = None,
                      exe: str | Path = DEFAULT_MASTER_EXE,
                      core: CoreParams = DEFAULT_CORE,
                      max_cycles: int = 16, consecutive: int = 2,
                      tolerances=None, enable_pin_burnup: bool = True,
                      hm_mtu: float | None = None,
                      timeout_s: float = 3600.0,
                      keep_work: bool = False,
                      purge_intermediate: bool = True,
                      cy1_cap_efpd: float | None = None) -> BootstrapResult:
    """Bootstrap the ``pair``/``feed`` cell to equilibrium; save the band seed.

    ``pkg_dir`` must already hold ``lib/MAS_XSL`` + ``lib/MAS_HFF``.  ``pair`` is
    ``"<a>_<b>"`` (or ``"<a>_<b>_<c>[...]"`` for a graded 3..5-type case) of
    library aliases.  Returns a :class:`BootstrapResult`
    (``cycles_needed`` = cy1 + reload cycles).

    ``cy1_cap_efpd`` caps the throwaway cy1 fresh-core cycle (see
    :func:`~lpopt.design.coredeck.build_cycle1_deck`).  ``None`` = the historical
    natural-EOC cy1.  Set it to the equilibrium cycle length ``2*B1/(241/feed+1)``
    so cy02 starts from an equilibrium-like carryover instead of a 34-37
    MWd/kgHM all-fresh discharge; the reload cycles are NOT capped, so the
    converged equilibrium the chain reports is unchanged.
    """
    pkg = Path(pkg_dir)
    xsl, hff = pkg / "lib" / "MAS_XSL", pkg / "lib" / "MAS_HFF"
    if not xsl.is_file() or not hff.is_file():
        raise BootstrapError(f"package {pkg} has no lib/MAS_XSL or lib/MAS_HFF")
    if aliases is None:
        aliases = library_aliases(pkg)
    try:
        members = case_batches(pair)          # 2 aliases (pair) or 3..5 (graded)
    except Exception as error:                # noqa: BLE001 — re-raise as our own
        raise BootstrapError(
            f"pair must be '<a>_<b>[_<c>...]' (2..{MAX_FRESH_TYPES} aliases), "
            f"got {pair!r}") from error
    for x in members:
        if x not in aliases:
            raise BootstrapError(f"pair type {x!r} not in library aliases {aliases}")

    key = CaseKey(pair, int(feed))
    work_root = pkg / "bootstrap_work" / key.folder
    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    _bootstrap_ok = False

    result = BootstrapResult(pair=pair, feed=int(feed), folder=key.folder,
                             converged=False, converged_at_cap=False, n_cycles=0,
                             cycles_needed=0, restart_path=None,
                             tolerance_margin=None, wall_s=0.0)
    try:
        # 1. cy1 fresh core -> first restart
        cy1 = build_cycle1_deck(aliases, members, core=core,
                                cap_efpd=cy1_cap_efpd)
        cy1_rst = run_cycle1(cy1, xsl, hff, exe, work_root / "cy1",
                             timeout_s=timeout_s)

        # 2. reload template referencing the cy1 restart basename
        n_units = fresh_units_from_feed(int(feed))
        pattern = random_genome(rng, pair, n_units).to_pattern()
        seed_dir = pkg / "cores" / key.folder / "bootstrap"
        seed_dir.mkdir(parents=True, exist_ok=True)
        reload_deck = build_reload_deck(aliases, cy1_rst.name, 2, core=core)
        template_path = seed_dir / "MAS_INP_cy02.inp"
        template_path.write_text(reload_deck, encoding="utf-8")

        # 3. drive the equilibrium chain
        master = MasterRunner(pkg, str(exe), work_root=work_root / "master",
                              timeout=timeout_s, keep_success=True)
        tol = (tolerances if tolerances is not None
               else EquilibriumTolerances())
        # keep_success=True keeps the FINAL cycle's dir (its MAS_RST is copied out
        # to bases/ below); purge_intermediate deletes the earlier cycles per-cycle
        # so the chain's peak footprint stays ~2 dirs, not max_cycles (USER DIRECTIVE).
        runner = PurgingEquilibriumRunner(master, max_cycles=max_cycles,
                                          consecutive=consecutive, tolerances=tol,
                                          keep_success=True,
                                          enable_pin_burnup=enable_pin_burnup,
                                          purge_intermediate=purge_intermediate)
        case_data = CaseData(key=key, cell=0.0, records=(),
                             template_path=template_path, restart_path=cy1_rst)
        eq = runner.run(case_data, pattern)

        # 4. harvest FOM + save the band seed restart
        result.converged = eq.converged
        result.converged_at_cap = eq.converged_at_cap
        result.n_cycles = eq.n_cycles
        result.cycles_needed = eq.n_cycles + 1        # + cy1
        result.tolerance_margin = eq.tolerance_margin
        fom = eq.fom
        if fom is not None:
            result.cyclen = fom.cyclen
            result.f_r = fom.f_r
            result.f_q = fom.f_q
            result.cbc_max = fom.cbc_max
            result.ao_abs = fom.ao_abs
            result.max_assembly_burnup = fom.max_assembly_burnup
            result.max_pin_burnup = fom.max_pin_burnup
            if result.cyclen is not None:
                mass = hm_mtu if hm_mtu is not None else core.hm_mtu
                result.discharge_burnup = estimate_discharge_burnup(
                    result.cyclen, int(feed), power_mw=core.power_mw, hm_mtu=mass)

        final = eq.cycles[-1].restart_path if eq.cycles else None
        if final is not None and final.is_file():
            bases = pkg / "bases" / key.folder
            bases.mkdir(parents=True, exist_ok=True)
            dst = bases / final.name
            shutil.copyfile(final, dst)
            result.restart_path = dst
        result.wall_s = time.monotonic() - t0
        _bootstrap_ok = True
    except Exception as exc:                            # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.wall_s = time.monotonic() - t0
    finally:
        # Purge only on SUCCESS.  On the exception path the work dir is the
        # ONLY evidence (MasterRunError's message even names it as "retained"
        # -- master.py:61-64), and an unconditional rmtree here destroyed the
        # T3_T4 cy02-hang forensics on 2026-08-11: the error pointed at a
        # directory this line had already deleted.  A failed bootstrap keeps
        # its work dir; the next run of the same pair clears it anyway
        # (rmtree at the start of make_band_restart).
        # _bootstrap_ok (not result.error) decides: `except Exception` misses
        # BaseException unwinds (KeyboardInterrupt/SystemExit during the
        # multi-hour MASTER subprocess), which would leave error=None and purge
        # the evidence anyway (ECC review 2026-08-12).
        # `result.converged` extends the same principle to the NO-EXCEPTION
        # failure: a chain that ran clean to max_cycles without settling is a
        # failed bootstrap too, and its per-cycle FOM trail is the only evidence
        # of WHY it never settled (ECC audit 2026-08-12).
        if not keep_work and _bootstrap_ok and result.converged:
            shutil.rmtree(work_root, ignore_errors=True)
    return result


__all__ = [
    "BootstrapError",
    "BootstrapResult",
    "DEFAULT_MASTER_EXE",
    "estimate_discharge_burnup",
    "library_aliases",
    "make_band_restart",
    "run_cycle1",
]
