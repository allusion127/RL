"""MASTER quarter-core deck synthesis for a paramA library (plan 12.1).

A self-contained port of ``2_LP/MOCHA/master_io.py`` (``build_cycle1_input`` /
``build_restart_input``) that emits decks the vendor harness parses byte-for-byte
(verified against the real ga80 ``cores/*/MAS_INP_cy12.inp``):

  * ``build_cycle1_deck`` — %LPD_BCH fresh core (bootstrap seed, no restart).
  * ``build_reload_deck`` — %LPD_SHF reload template (irrst=1); a placeholder
    9-line SHF is later overwritten by ``replace_lpd_shf`` and its restart
    reference by ``advance_cycle_deck`` (harness contract).

Fuel "type names" here are the 2-char MASTER aliases (already MASTER-safe), used
directly as batch ids; each maps to the XS set ``FA_<alias>``.  Two adjustments
vs the MOCHA reference: (1) the batch id **is** the alias (no second remap), and
(2) the EOC ``%EDT_OPT`` ippi flag is left 0 so the vendor ``enable_pin_burnup``
path (``enable_ppi_output``) controls MAS_PPI, exactly like the ga80 template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..search.genome import MAX_FRESH_TYPES

# --------------------------------------------------------------------------- #
# geometry (ported verbatim from MOCHA geometry.py)
# --------------------------------------------------------------------------- #
COL_LABELS = ["RE", "A", "B", "C", "D", "E", "F", "G", "H", "J",
              "K", "L", "M", "N", "P", "R", "S", "T", "RE"]
ROW_LABELS = ["RE"] + [str(n) for n in range(1, 18)] + ["RE"]
CENTER = 10

_FULL_BCH_MAP_TEXT = """\
o   o   o   o   o   R3  R1  R1  R1  R1  R1  R1  R1  R3  o   o   o   o   o
o   o   o   R3  R1  R2  X1  X1  X1  X1  X1  X1  X1  R2  R1  R3  o   o   o
o   o   R3  R2  X1  X1  X1  X1  X0  X0  X0  X1  X1  X1  X1  R2  R3  o   o
o   R3  R2  X1  X1  X1  X0  X1  X0  X1  X0  X1  X0  X1  X1  X1  R2  R3  o
o   R1  X1  X1  X0  X0  X0  X1  X0  X0  X0  X1  X0  X0  X0  X1  X1  R1  o
R3  R2  X1  X1  X0  X1  X0  X0  X0  X0  X0  X0  X0  X1  X0  X1  X1  R2  R3
R1  X1  X1  X0  X0  X0  X0  X0  X1  X0  X1  X0  X0  X0  X0  X0  X1  X1  R1
R1  X1  X1  X1  X1  X0  X0  X0  X0  X0  X0  X0  X0  X0  X1  X1  X1  X1  R1
R1  X1  X0  X0  X0  X0  X1  X0  X0  X0  X0  X0  X1  X0  X0  X0  X0  X1  R1
R1  X1  X0  X1  X0  X0  X0  X0  X0  X0  X0  X0  X0  X0  X0  X1  X0  X1  R1
R1  X1  X0  X0  X0  X0  X1  X0  X0  X0  X0  X0  X1  X0  X0  X0  X0  X1  R1
R1  X1  X1  X1  X1  X0  X0  X0  X0  X0  X0  X0  X0  X0  X1  X1  X1  X1  R1
R1  X1  X1  X0  X0  X0  X0  X0  X1  X0  X1  X0  X0  X0  X0  X0  X1  X1  R1
R3  R2  X1  X1  X0  X1  X0  X0  X0  X0  X0  X0  X0  X1  X0  X1  X1  R2  R3
o   R1  X1  X1  X0  X0  X0  X1  X0  X0  X0  X1  X0  X0  X0  X1  X1  R1  o
o   R3  R2  X1  X1  X1  X0  X1  X0  X1  X0  X1  X0  X1  X1  X1  R2  R3  o
o   o   R3  R2  X1  X1  X1  X1  X0  X0  X0  X1  X1  X1  X1  R2  R3  o   o
o   o   o   R3  R1  R2  X1  X1  X1  X1  X1  X1  X1  R2  R1  R3  o   o   o
o   o   o   o   o   R3  R1  R1  R1  R1  R1  R1  R1  R3  o   o   o   o   o
"""

FULL_BCH_MAP = [line.split() for line in _FULL_BCH_MAP_TEXT.strip().splitlines()]
QUARTER_BCH_MAP = [row[CENTER - 1:] for row in FULL_BCH_MAP[CENTER - 1:]]


def is_fuel(token: str) -> bool:
    return token != "o" and not token.startswith("R")


REFLECTOR_BATCHES = (("R1", -2), ("R2", -3), ("R3", -4))
REFLECTOR_COMPS = (
    (-1, "REF_AXIAL_B"), (-2, "REF_R1"), (-3, "REF_R2"),
    (-4, "REF_R3"), (-5, "REF_AXIAL_T"),
)


@dataclass
class CoreParams:
    """APR1400 quarter-core physics constants (MOCHA CoreConfig defaults)."""

    plant_id: str = "APRQ"
    title: str = "Quarter Core Depletion for APR1400"
    power_mw: float = 3983.0
    tin: float = 290.6
    trise: float = 33.3
    tavg: float = 307.25
    press: float = 155.132
    mflow: float = 3480.0
    nz: int = 27
    zmesh: tuple[str, ...] = ("30.0", "25*15.24", "30.0")
    wide: float = 20.7772
    height: float = 381.0
    # depletion
    eoc_boron_ppm: float = 10.0
    pool_decay_days: float = 60.0
    initial_steps: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)
    adaptive_step: float = -30.0
    #: Full-core HM (heavy-metal / U) mass [MTU], used only for the derived
    #: discharge-burnup estimate (plan 12.3).  MASTER's %GEN_THD power (3983 MW)
    #: is the FULL-core rated thermal power, so the burnup basis is full-core too:
    #: 241 FA * ~0.435 MTU/FA ~ 104.8 MTU.  (The DeCART MASS(g) inventory is a
    #: per-unit-length lattice quantity and is NOT used for this absolute mass.)
    hm_mtu: float = 104.8


DEFAULT_CORE = CoreParams()


# --------------------------------------------------------------------------- #
# card fragments
# --------------------------------------------------------------------------- #
def _dims(n_types: int) -> tuple[int, int]:
    """(nbatch, ncomp) for a paramA library of ``n_types`` fuel sets."""
    return len(REFLECTOR_BATCHES) + n_types, len(REFLECTOR_COMPS) + n_types


def _job_common(core: CoreParams, icycle: int) -> str:
    return f"""%JOB_VER
        4.0     2                                                               # xform, ichntyp
%JOB_TIT
        {core.title} CY{icycle:02d}
%JOB_IDE
        {core.plant_id}    {icycle}                                                               # iplant, icycle
%JOB_MDL
        2       2       0       2       0                                       # irod, iextc, ixsmod, iloc, idch,
        0       0                                                               # iben, idete
"""


def _gen_pin_edt() -> str:
    return """%GEN_PIN
        1       2       16      236                                             # icornf, iweigh, npin, nfrod
%EDT_PIN
        3       1                                                               # iprloc, ipin,
        0       0       0       0       0       0                               # ixb, ixe, jyb, jye, kzb, kze
%EDT_OUT
        1       0       1       0       1                                       # ittabl, iusdck, iprerr, iprusd, iprmap,
        0       0       1       1       0                                       # iprxs, iprcur, iprflx, iprpff, iprcon,
        1       0       1       1       0                                       # iprtim, iprth, iprfb, iprbun, iprdet
"""


def _gen_dim(core: CoreParams, n_types: int) -> str:
    n_batch, n_comp = _dims(n_types)
    return f"""%GEN_DIM
        10      10      {core.nz}      {n_batch}       {n_comp}                                      # nx, ny, nz, nbatch, ncomp,
        3       4       1       2       1                                       # ndim, ngeo, nsym, ndivxy, ndivz,
        2                                                                       # ng
"""


def _gen_geo_sym_cdn(core: CoreParams) -> str:
    zmesh = "\n".join(f"        {z}" for z in core.zmesh)
    cols = "  ".join(f"{c:<3}" for c in COL_LABELS).rstrip()
    rows = "  ".join(f"{r:<3}" for r in ROW_LABELS).rstrip()
    return f"""%GEN_GEO
        {core.wide}  {core.height}                                                            # wide, height,
{zmesh}
%GEN_SYM
        -1      -1      0                                                       # isymlx, isymly, isymlz,
        0       0       0                                                       # isymrx, isymry, isymrz
%GEN_CDN
        {cols}      # extx(j), j=1,nx (full-core labels)
        {rows}      # exty(j), j=1,ny (full-core labels)
"""


def _lpd_static(aliases: list[str]) -> str:
    """%LPD_B&C / %LPD_C&X / %LPD_HFF for reflectors + fuel aliases."""
    lines = ["%LPD_B&C"]
    for batch, comp in REFLECTOR_BATCHES:
        lines.append(f"        {batch:<7}                 27*{comp}")
    for idx, alias in enumerate(aliases, start=1):
        lines.append(f"        {alias:<7} -1              25*{idx:<12} -5")

    lines.extend(["", "%LPD_C&X"])
    for comp, xs_name in REFLECTOR_COMPS:
        lines.append(f"        {comp:<7} {xs_name:<15} 0")
    for idx, alias in enumerate(aliases, start=1):
        lines.append(f"        {idx:<7} FA_{alias:<12} 0")

    lines.extend(["", "%LPD_HFF"])
    for idx, alias in enumerate(aliases, start=1):
        lines.append(f"        {idx:<7} FA_{alias}")
    return "\n".join(lines) + "\n"


def _capped_dep_ramp(core: CoreParams, cap_efpd: float) -> str:
    """Fixed-length ``%EXE_DEP`` steps from the ``initial_steps`` end to ``cap_efpd``.

    Used instead of the adaptive natural-EOC block when the caller caps cy1.  Only
    the positive-``delt`` idiom already used by ``initial_steps`` is emitted — no
    ``tgobj`` search — so the deck stays inside syntax MASTER is known to accept.
    """
    reached = float(sum(core.initial_steps))
    remaining = round(float(cap_efpd) - reached, 6)
    if remaining <= 0.0:
        raise ValueError(
            f"cap_efpd {cap_efpd:g} must exceed the initial_steps total "
            f"{reached:g} EFPD")
    step = abs(float(core.adaptive_step)) or 30.0
    out = []
    while remaining > 1e-6:
        delt = step if remaining > step + 1e-6 else remaining
        out.append(f"""%EXE_DEP
        {delt:<7g} 0                                                               # delt, itg
/
""")
        remaining = round(remaining - delt, 6)
    return "".join(out)


def _exe_blocks(core: CoreParams, *, write_boc_rst: bool = False,
                write_eoc_rst: bool = True,
                cap_efpd: float | None = None) -> str:
    out = ["""%EXE_STD                                                                        # HFP, Eq. Xenon
        boron   eq      tr      1.0                                             # isearch, ixe, ism, pload
"""]
    for k, delt in enumerate(core.initial_steps):
        out.append(f"""%EXE_DEP
        {delt:<7g} 0                                                               # delt, itg
""")
        if k == 0:
            iwrst = 1 if write_boc_rst else 0
            out.append(f"""%EDT_OPT
        {iwrst:<7d} 0       0       0                                               # iwrst, icob, ippi, icmp
""")
        out.append("/\n")
    if cap_efpd is None:
        out.append(f"""%EXE_DEP                                                                        # natural EOC
        {core.adaptive_step:<7g} 0                                                               # delt, itg
        boron   {core.eoc_boron_ppm:g}                                                              # tgobj, tgval
/
""")
    else:
        out.append(f"""#  cy1 CAPPED at {float(cap_efpd):g} EFPD -- the adaptive natural-EOC search
#  is replaced by fixed steps.  The restart below is the band seed, so its
#  burnup must look like ONE equilibrium cycle, not an all-fresh core's full
#  natural cycle.
""")
        out.append(_capped_dep_ramp(core, cap_efpd))
    out.append(f"""%EXE_STD
        keff    tr      tr      1.0                                             # isearch, ixe, ism, pload
%EDT_OPT
        {1 if write_eoc_rst else 0:<7d} 0       0       0                                               # iwrst, icob, ippi(0; enable_ppi_output toggles), icmp
/
END
""")
    return "".join(out)


# --------------------------------------------------------------------------- #
# fresh-core batch map (%LPD_BCH) + placeholder loading (%LPD_SHF)
# --------------------------------------------------------------------------- #
def _lpd_bch_quarter(fresh_types: Sequence[str]) -> str:
    """Fresh-core %LPD_BCH map: legacy tokens X0/X1 -> the fresh aliases.

    Two types is the historical mapping (``X0 -> a``, ``X1 -> b``) and is emitted
    byte-for-byte.  A graded bootstrap core (3..:data:`MAX_FRESH_TYPES` types)
    keeps ``X0 -> a`` and **round-robins** the ``X1`` cells over ``types[1:]`` in
    row-major order, so every non-first type is spread over the same legacy zone
    rather than concentrated in one annulus — a deterministic, restart-only seed
    map.  The round-robin reduces to "always ``b``" for two types and to the
    ``b/c`` alternation for three, so both earlier decks stay byte-identical.
    (The *searched* loading is the %LPD_SHF body, which carries a per-position
    type and needs no change at all; this map only sets the cycle-1 bootstrap
    composition.)
    """
    types = list(fresh_types)
    if not (2 <= len(types) <= MAX_FRESH_TYPES):
        raise ValueError(
            f"a fresh-core map needs 2..{MAX_FRESH_TYPES} fresh types, "
            f"got {types}")
    a = types[0]
    rest = types[1:]
    lines = ["%LPD_BCH                                                                        # quarter core (SE quadrant incl. center)"]
    x1_seen = 0
    for row in QUARTER_BCH_MAP:
        out = []
        for token in row:
            if token == "X0":
                out.append(a)
            elif token == "X1":
                out.append(rest[x1_seen % len(rest)])
                x1_seen += 1
            else:
                out.append(token)
        lines.append("        " + "  ".join(f"{t:<2}" for t in out).rstrip())
    return "\n".join(lines) + "\n"


def placeholder_shf(alias: str) -> str:
    """A structurally valid 9-line quarter %LPD_SHF body (all-fresh of ``alias``).

    Content is irrelevant — the harness overwrites it via ``replace_lpd_shf`` —
    but the shape (9 non-blank, no ``%`` cards) matches the vendor contract.
    """
    lines = []
    for row in QUARTER_BCH_MAP:
        n = sum(1 for t in row if is_fuel(t))
        if n == 0:
            continue
        entries = [f"F {alias:<2}   0"] * n
        lines.append("        " + ", ".join(entries) + ",")
    return "\n".join(lines) + "\n"


def _lpd_shf_block(shf_body: str) -> str:
    return "%LPD_SHF                                                                        # ishuff(i,j): fuel cells only\n" + shf_body


# --------------------------------------------------------------------------- #
# public builders
# --------------------------------------------------------------------------- #
def build_cycle1_deck(aliases: list[str], fresh_pair: Sequence[str],
                      *, core: CoreParams = DEFAULT_CORE,
                      write_eoc_rst: bool = True,
                      cap_efpd: float | None = None) -> str:
    """Initial quarter-core fresh-core deck (bootstrap seed, no restart).

    ``fresh_pair`` is the case's fresh-type alphabet: a 2-tuple (historical, the
    deck is byte-identical) or a 3..5-tuple for a graded case
    (:func:`_lpd_bch_quarter`).

    ``cap_efpd`` stops cy1 after a fixed number of EFPD instead of at its natural
    EOC (``boron`` -> ``core.eoc_boron_ppm``).  ``None`` (default) keeps the
    historical natural-EOC behaviour byte-for-byte.

    Why cap: an ALL-FRESH core runs far longer than any equilibrium cycle
    (measured 894 EFPD for T3_T4, 981 for T5_T6 == 34-37 MWd/kgHM), so an
    uncapped cy1 hands cy02 a carryover batch ~1.5-2.0x deeper than the
    equilibrium once-burned batch it is supposed to represent.  The principled
    value is the equilibrium cycle burnup from the linear reactivity model,

        Bc = 2 * B1 / (n + 1),    n = 241 / feed  (residence in cycles)

    with ``B1`` the natural cy1 burnup — i.e. cap so the carryover IS one
    equilibrium cycle deep.  (Validated: T5_T6 @ feed 121 gives Bc = 655.8 EFPD
    against a measured equilibrium cyclen of 643.6, +1.9 %.)
    """
    if not aliases:
        raise ValueError("at least one fuel alias is required")
    text = f"""%JOB_TYP
        0       stead                                                           # irrst, jobtyp
        xsl     MAS_XSL
        hff     MAS_HFF
        out     MAS_OUT
        sum     MAS_SUM
{_job_common(core, 1)}%GEN_MTH
        9       1       0       3       1                                       # imeth, itral, ibndc, ibupco, iasex,
        1       1       1       1       -1                                      # imlc, istocl, imom, ibal, faccmr,
        12      1                                                               # isolth, idepl
%GEN_LMT
        1.E-5   200     1       1       1                                       # epsflx, ncycle, nfine, icrnth, icrncs,
        4000    -4000                                                           # upppm, loppm
%GEN_THD
        {core.power_mw}    {core.tin}  {core.trise}   {core.tavg}  {core.press}                                   # power, tin, trise, tavg, press,
        {core.mflow}    1.0                                             # mflow, hgfl
{_gen_pin_edt()}{_gen_dim(core, len(aliases))}{_gen_geo_sym_cdn(core)}{_lpd_bch_quarter(fresh_pair)}{_lpd_static(aliases)}#################################################################################
{_exe_blocks(core, write_eoc_rst=write_eoc_rst, cap_efpd=cap_efpd)}"""
    return text


def build_reload_deck(aliases: list[str], restart_basename: str, icycle: int,
                      *, core: CoreParams = DEFAULT_CORE,
                      shf_body: str | None = None,
                      write_eoc_rst: bool = True) -> str:
    """Reload template: reads ``restart_basename``, applies a %LPD_SHF loading.

    ``shf_body`` is the 9-line quarter SHF body; when omitted an all-fresh
    placeholder (to be replaced by the harness) is used.  Produces exactly the
    reload idiom ``validate_reload_deck`` accepts (irrst=1, one %LPD_SHF, no
    %LPD_BCH, GEN_DIM nbatch/ncomp = (3+N, 5+N)).
    """
    if not aliases:
        raise ValueError("at least one fuel alias is required")
    if not restart_basename.upper().startswith("MAS_RST."):
        raise ValueError("restart_basename must be a MAS_RST.* name")
    body = shf_body if shf_body is not None else placeholder_shf(aliases[0])
    text = f"""%JOB_TYP
1       stead                                                           # NUMBER OF RESTART FILE
        {restart_basename}
        xsl     MAS_XSL
        hff     MAS_HFF
        out     MAS_OUT
        sum     MAS_SUM
{_job_common(core, icycle)}{_gen_pin_edt()}{_gen_dim(core, len(aliases))}%LPD_PUL
        1       {core.pool_decay_days:g}                                                              # nrst, pooltim
{_lpd_shf_block(body)}{_lpd_static(aliases)}#################################################################################
{_exe_blocks(core, write_eoc_rst=write_eoc_rst)}"""
    return text


def library_dims(n_types: int) -> tuple[int, int]:
    """(nbatch, ncomp) for the WaveVerifier ``library_dims`` gate."""
    return _dims(n_types)


__all__ = [
    "CoreParams",
    "DEFAULT_CORE",
    "REFLECTOR_BATCHES",
    "REFLECTOR_COMPS",
    "build_cycle1_deck",
    "build_reload_deck",
    "library_dims",
    "placeholder_shf",
]
