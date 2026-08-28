"""Physics-based fuel feature table (plan section 4.3).

``build_fuel_table(cfg_or_paths)`` assembles one row per ``(library_id,
type_id)`` and persists it to ``data/store/fuel_types.parquet``.  ``(library_id,
type_id)`` is the key: the *same* type id (e.g. ``"B1"``) intentionally appears
in several libraries with different physics, so the library is mandatory.

Sources, in the priority order of plan 4.3 — each is a separate function that
tags the row it feeds via ``source_flags``:

  1. ``FA_*.out`` MASS(g)     -> ``u_avg_enrichment`` / ``u_mass_g``
     (``parse_fa_mass_out`` — ported from ``2_LP/MOCHA/config.py`` ~L855-925,
     ``parse_decart_fresh_inventory`` / ``discover_fuel_data_from_fa_dir``).
  2. ``dec_FA_*.inp`` MATERIAL + directory name ``{gd_wt}_{n_gd}_z{1,2}``
     -> ``enr_main`` (5.8) / ``enr_zone`` (5.1) / ``gd_u_enr`` (4.0) / ``n_gd``
     / ``gd_wt`` / ``axial_zone`` (``parse_dec_material`` + ``parse_dir_geometry``).
  3. HGC ``%DIST`` Gd-pin heuristic (BOC relative power < 0.6) -> ``n_gd`` for
     the letter-type ga80 library (``count_gd_pins_from_hgc``).  These HGCs are
     frequently OneDrive-dehydrated (readable -> parse, unreadable -> fall
     through to the manual anchor with ``feature_poor=True``).
  4. Legacy A0/A1 (hard-coded from ``2_LP/MOCHA/config.py`` ~L78-81) -> library
     ``legacy_a`` with ``source_flags=["mocha_hardcoded"]``.
  5. Manual fallback ``config/fuel_types_manual.yaml`` -> ga80 letter anchors.

The k-inf curve columns (``kinf0 kinf10 kinf20 kinf30 bu_k1``) are left NaN for
now; they are populated in M2 from EDIT5.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
#: Columns of the persisted table, in order.
SCHEMA_COLUMNS: list[str] = [
    "library_id",
    "type_id",
    "u_avg_enrichment",
    "enr_main",
    "enr_zone",
    "u_mass_g",
    "n_gd",
    "gd_wt",
    "gd_u_enr",
    "axial_zone",
    "source_flags",
    "feature_poor",
    # k-inf curve characteristic points — reference-depletion k-inf(BU) curve
    # sampled at 0/10/20/30 GWd/tU + burnup where k-inf crosses 1.0 (cond_v4).
    "kinf0",
    "kinf10",
    "kinf20",
    "kinf30",
    "bu_k1",
    # cond_v4 physics features (NaN when unharvested).  BOC pin-power peaking,
    # 2-group BOC macro cross sections, BOC assembly discontinuity factors, BOC
    # reactivity coefficients, and the zoning-pin census.
    "ff_pin_max",
    "xs_d1", "xs_d2", "xs_a1", "xs_a2", "xs_nf1", "xs_nf2", "xs_s12",
    "adf_face_g1", "adf_face_g2", "adf_corner_g1", "adf_corner_g2",
    "boron_worth", "doppler_coef", "mtc_dmod", "cr1_worth",
    "zone_pin_count",
    # pin-burnup peaking-ratio curve summary (BRP/BU vs BU) — the physics backbone
    # of the serve-side pin-burnup estimator (lpopt.model.pinbu_physics).  Harvested
    # from .sum EDIT3 (BRP column) with an HGC %DIST map7 fallback for ga80.
    "pin_bu_r_inf", "pin_bu_paramA", "pin_bu_ratio_asym", "pin_bu_bu_max",
    "pin_bu_n_pts",
    # pin-cell geometry harvested from the dec inp GEOM block (pitch_radius_readiness
    # sec. 4a).  NaN for HGC-only types (ga80/legacy ship no dec inp) — same NaN
    # contract as ``zone_pin_count``.  These are CONSTANT across every training
    # lattice today (pin_pitch 1.285, asm_pitch 20.7772, pellet 0.4096, clad
    # 0.4178/0.4750); harvesting them makes the population range explicit so the
    # OOD guard can flag a future pin-pitch/pin-radius variant, and gives a v5
    # geometry-aware featurizer an input channel.  ``asm_pitch`` is the frozen
    # assembly-envelope anchor (== coredeck CoreParams.wide) — the load-bearing
    # constraint that keeps the MASTER core model valid (sec. 0 of the review).
    "pin_pitch", "asm_pitch", "r_pellet", "r_clad_in", "r_clad_out",
    "p_over_d", "v_mod_over_v_fuel",
    # k-conv (k-inf vs burnup) CURVE-SHAPE features harvested from the SAME
    # reference-depletion curve as kinf0..kinf30 (.sum EDIT2 K-CONV rows / HGC
    # %TITL k-inf per state).  These summarise the burnable-absorber holddown ->
    # release SIGNATURE in reactivity space (rho = (k-1)/k*1e5 pcm) independently
    # of WHICH absorber produced it (Gd today; IFBA/Er/Dy tomorrow) — the shape,
    # not the poison design, is the universal, poison-agnostic feature.  Additive
    # nullable floats (NaN when the curve is unharvested; see KCONV_SHAPE_COLUMNS).
    "kinf_dip", "bu_dip_gwd", "kinf_peak", "bu_peak_gwd",
    "reactivity_swing_pcm", "rho_boc_minus_peak_pcm",
    "depletion_slope_pcm_per_gwd", "kinf_eol50", "kconv_is_monotone",
]

#: cond_v4 columns appended to the schema — every one is a nullable float.
COND_V4_COLUMNS: tuple[str, ...] = (
    "ff_pin_max",
    "xs_d1", "xs_d2", "xs_a1", "xs_a2", "xs_nf1", "xs_nf2", "xs_s12",
    "adf_face_g1", "adf_face_g2", "adf_corner_g1", "adf_corner_g2",
    "boron_worth", "doppler_coef", "mtc_dmod", "cr1_worth",
    "zone_pin_count",
)

#: pin-burnup peaking-ratio curve summary columns (additive; all nullable float).
#: ``ratio(BU) = BRP/BU`` is the peak-pin-to-assembly-average burnup ratio at each
#: depletion state; it declines from the BOC power form factor (``ff_pin_max``)
#: toward a high-burnup plateau.  We persist a compact summary: an auto-fit
#: ``ratio(BU) = r_inf + paramA/BU`` over the discharge tail (``paramA`` auto), the
#: directly-observed plateau ratio at the last state, the max state burnup, and the
#: point count (QC).  ``ff_pin_max`` (already stored) anchors the BU->0 limit.
PIN_BU_COLUMNS: tuple[str, ...] = (
    "pin_bu_r_inf", "pin_bu_paramA", "pin_bu_ratio_asym", "pin_bu_bu_max",
    "pin_bu_n_pts",
)

#: pin-cell geometry columns (additive; all nullable float) harvested from the dec
#: inp ``GEOM`` block (pitch_radius_readiness.md sec. 4a).  ``pin_pitch``/``asm_pitch``
#: come from the ``pitch`` card (token1 / token2); ``r_pellet``/``r_clad_in``/
#: ``r_clad_out`` from the ``cellgeo 1`` leading radii; ``p_over_d`` and
#: ``v_mod_over_v_fuel`` are the derived moderation ratios (single-cell, guide tubes
#: ignored).  NaN for any type with no dec inp (ga80/legacy).
GEOM_COLUMNS: tuple[str, ...] = (
    "pin_pitch", "asm_pitch", "r_pellet", "r_clad_in", "r_clad_out",
    "p_over_d", "v_mod_over_v_fuel",
)

#: k-conv curve-shape columns (additive; all nullable float) computed from the
#: reference k-inf(BU) curve by :func:`kconv_curve_shape` (shared by the .sum and
#: HGC parsers via :func:`_curve_and_coeffs`, so a type carrying both products
#: agrees by construction).  ``kinf_dip``/``bu_dip_gwd`` mark the burnable-absorber
#: suppression trough, ``kinf_peak``/``bu_peak_gwd`` the post-dip burnout hump;
#: ``reactivity_swing_pcm`` = rho_peak-rho_dip (holddown release), and
#: ``depletion_slope_pcm_per_gwd`` the post-peak reactivity decay rate.  These are
#: the POISON-AGNOSTIC curve signature intended to become v5 model channels (the
#: adopted physics prior consumes ``reactivity_swing_pcm`` + ``depletion_slope_*``
#: directly).  ``kinf_dip``/``bu_dip_gwd``/``reactivity_swing_pcm`` are NaN for a
#: near-monotone curve (no prominent hump), flagged by ``kconv_is_monotone``.
KCONV_SHAPE_COLUMNS: tuple[str, ...] = (
    "kinf_dip", "bu_dip_gwd", "kinf_peak", "bu_peak_gwd",
    "reactivity_swing_pcm", "rho_boc_minus_peak_pcm",
    "depletion_slope_pcm_per_gwd", "kinf_eol50", "kconv_is_monotone",
)

_FLOAT_COLUMNS = (
    "u_avg_enrichment", "enr_main", "enr_zone", "u_mass_g", "gd_wt", "gd_u_enr",
    "kinf0", "kinf10", "kinf20", "kinf30", "bu_k1",
    *COND_V4_COLUMNS,
    *PIN_BU_COLUMNS,
    *GEOM_COLUMNS,
    *KCONV_SHAPE_COLUMNS,
)


@dataclass
class FuelVec:
    """One ``(library_id, type_id)`` fuel-feature row."""

    library_id: str
    type_id: str
    u_avg_enrichment: float | None = None
    enr_main: float | None = None
    enr_zone: float | None = None
    u_mass_g: float | None = None
    n_gd: int | None = None
    gd_wt: float | None = None
    gd_u_enr: float | None = None
    axial_zone: str | None = None            # "z1" | "z2" | None
    source_flags: list[str] = field(default_factory=list)
    feature_poor: bool = False
    kinf0: float | None = None
    kinf10: float | None = None
    kinf20: float | None = None
    kinf30: float | None = None
    bu_k1: float | None = None
    # cond_v4 physics features (see COND_V4_COLUMNS / SHARED COLUMN CONTRACT).
    ff_pin_max: float | None = None
    xs_d1: float | None = None
    xs_d2: float | None = None
    xs_a1: float | None = None
    xs_a2: float | None = None
    xs_nf1: float | None = None
    xs_nf2: float | None = None
    xs_s12: float | None = None
    adf_face_g1: float | None = None
    adf_face_g2: float | None = None
    adf_corner_g1: float | None = None
    adf_corner_g2: float | None = None
    boron_worth: float | None = None
    doppler_coef: float | None = None
    mtc_dmod: float | None = None
    cr1_worth: float | None = None
    zone_pin_count: float | None = None
    # pin-burnup peaking-ratio curve summary (see PIN_BU_COLUMNS).
    pin_bu_r_inf: float | None = None
    pin_bu_paramA: float | None = None
    pin_bu_ratio_asym: float | None = None
    pin_bu_bu_max: float | None = None
    pin_bu_n_pts: float | None = None
    # pin-cell geometry (see GEOM_COLUMNS).
    pin_pitch: float | None = None
    asm_pitch: float | None = None
    r_pellet: float | None = None
    r_clad_in: float | None = None
    r_clad_out: float | None = None
    p_over_d: float | None = None
    v_mod_over_v_fuel: float | None = None
    # k-conv curve-shape features (see KCONV_SHAPE_COLUMNS).
    kinf_dip: float | None = None
    bu_dip_gwd: float | None = None
    kinf_peak: float | None = None
    bu_peak_gwd: float | None = None
    reactivity_swing_pcm: float | None = None
    rho_boc_minus_peak_pcm: float | None = None
    depletion_slope_pcm_per_gwd: float | None = None
    kinf_eol50: float | None = None
    kconv_is_monotone: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.library_id, self.type_id)

    def to_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in _FLOAT_COLUMNS:
                rec[f.name] = float("nan") if val is None else float(val)
            elif f.name == "source_flags":
                rec[f.name] = list(val)
            else:
                rec[f.name] = val
        return rec


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
@dataclass
class FuelPaths:
    """Resolved (absolute) source/output paths for the fuel table."""

    apr1400_root: Path       # holds 260624/, 5.8_5.1/, CPHA/ subtrees
    ga80_hgc: Path           # FEASIBLE_PACKAGE/hgc (letter-type HGCs)
    manual_yaml: Path        # config/fuel_types_manual.yaml
    store: Path              # output parquet
    #: Optional paramA design directory (plan 12): holds ``designs.json`` + the
    #: ``FA_<alias>.out`` inventories of freshly generated lattices.  ``None``
    #: (default) leaves the pre-existing five libraries untouched.
    paramA_root: Path | None = None


def _resolve(p: str | Path, base: Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q)


def fuel_paths_from_config(cfg: Any) -> FuelPaths:
    """Build :class:`FuelPaths` from an :class:`lpopt.config.LpoptConfig`."""
    base = cfg.source_path.parent if getattr(cfg, "source_path", None) else Path.cwd()
    f = cfg.fuel
    paramA = None
    design = getattr(cfg, "design", None)
    if design is not None and getattr(design, "paramA_root", None):
        cand = _resolve(design.paramA_root, base)
        if cand.exists():
            paramA = cand
    return FuelPaths(
        apr1400_root=_resolve(f.apr1400_root, base),
        ga80_hgc=_resolve(f.ga80_hgc, base),
        manual_yaml=_resolve(f.manual_yaml, base),
        store=_resolve(f.store, base),
        paramA_root=paramA,
    )


def _as_fuel_paths(cfg_or_paths: Any) -> FuelPaths:
    if isinstance(cfg_or_paths, FuelPaths):
        return cfg_or_paths
    if isinstance(cfg_or_paths, dict):
        pa = cfg_or_paths.get("paramA_root")
        return FuelPaths(
            apr1400_root=Path(cfg_or_paths["apr1400_root"]),
            ga80_hgc=Path(cfg_or_paths["ga80_hgc"]),
            manual_yaml=Path(cfg_or_paths["manual_yaml"]),
            store=Path(cfg_or_paths["store"]),
            paramA_root=Path(pa) if pa else None,
        )
    if hasattr(cfg_or_paths, "fuel"):          # LpoptConfig (duck-typed)
        return fuel_paths_from_config(cfg_or_paths)
    raise TypeError(
        "build_fuel_table expects an LpoptConfig, a FuelPaths, or a dict of "
        f"paths; got {type(cfg_or_paths).__name__}"
    )


def _read_text_flex(path: Path) -> str:
    """Read text with utf-8-sig, falling back to cp949 (Windows/Korean files)."""
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# source 1: FA_*.out MASS(g)  (ported from MOCHA config.py L855-925)
# --------------------------------------------------------------------------- #
_FA_OUT_RE = re.compile(r"^FA_(?P<name>.+)\.out$", re.I)
_FA_DEC_RE = re.compile(r"^dec_FA_(?P<name>.+)\.inp$", re.I)
_BURNUP_RE = re.compile(r"BURNUP\s*=\s*([+-]?[0-9.]+(?:[Ee][+-]?\d+)?)\s+MWD/KGHM")


def _read_first_burnup_block(path: Path) -> list[str]:
    """Stream a DeCART ``.out`` file only up to the *second* BURNUP marker.

    The BURNUP=0 fresh inventory sits near the top (~line 1800 of a ~27 MB
    file), so reading the whole file — as the reference MOCHA parser does — is
    wasteful.  Line indices in the returned list match the original file, so the
    downstream MOCHA logic is byte-for-byte identical to the reference.
    """
    lines: list[str] = []
    seen_zero = False
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            lines.append(line)
            m = _BURNUP_RE.search(line)
            if not m:
                continue
            if seen_zero:
                break                       # next BURNUP marker -> block boundary
            if abs(float(m.group(1))) < 1.0e-10:
                seen_zero = True
    return lines


def parse_fa_mass_out(path: str | Path) -> dict:
    """Parse the BURNUP=0 MASS(g) TOTAL row of a DeCART ``FA_*.out`` file.

    Ported from ``2_LP/MOCHA/config.py`` ``parse_decart_fresh_inventory``
    (~L855-925).  Returns U-235/U-238 TOTAL masses [g] and the U-mass-weighted
    fresh enrichment ``100*m235/(m235+m238)`` [w/o].
    """
    p = Path(path)
    lines = _read_first_burnup_block(p)

    burn0 = None
    for i, line in enumerate(lines):
        m = _BURNUP_RE.search(line)
        if m and abs(float(m.group(1))) < 1.0e-10:
            burn0 = i
            break
    if burn0 is None:
        raise ValueError(f"{p}: no BURNUP = 0.0000 block")

    end = len(lines)
    for j in range(burn0 + 1, len(lines)):
        if _BURNUP_RE.search(lines[j]):
            end = j
            break

    inv = None
    for j in range(burn0, end):
        if "ISOTOPE INVENTORY" in lines[j]:
            inv = j
            break
    if inv is None:
        raise ValueError(f"{p}: no ISOTOPE INVENTORY in BURNUP=0 block")

    in_mass = False
    isotopes = None
    total = None
    for j in range(inv, end):
        line = lines[j]
        toks = line.split()
        if "== MASS (g)" in line:
            in_mass = True
            continue
        if not in_mass:
            continue
        if line.startswith("=====") and "ISOTOPE INVENTORY" not in line:
            break
        if len(toks) > 3 and toks[0] == "ASY" and toks[1] == "PIN":
            isotopes = toks[2:]
            continue
        if toks and toks[0] == "TOTAL":
            total = toks[1:]
            break
    if isotopes is None or total is None:
        raise ValueError(f"{p}: no MASS(g) isotope header/TOTAL row")

    try:
        i235 = isotopes.index("92235")
        i238 = isotopes.index("92238")
        u235 = float(total[i235])
        u238 = float(total[i238])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{p}: cannot extract 92235/92238 TOTAL masses") from exc
    u_mass = u235 + u238
    if u_mass <= 0.0:
        raise ValueError(f"{p}: non-positive U mass from 92235/92238")
    return {
        "u235_g": u235,
        "u238_g": u238,
        "u_mass_g": u_mass,
        "u_avg_enrichment": 100.0 * u235 / u_mass,
        "source_out": str(p),
    }


# --------------------------------------------------------------------------- #
# source 2: dec_FA_*.inp MATERIAL block + directory name
# --------------------------------------------------------------------------- #
_DIR_GEOM_RE = re.compile(r"^(?P<gd_wt>\d+(?:\.\d+)?)_(?P<n_gd>\d+)_z(?P<zone>[12])$")


def parse_dir_geometry(dir_name: str) -> dict:
    """Parse a lattice directory name ``{gd_wt}_{n_gd}_z{1,2}`` (e.g. ``6_20_z1``)."""
    m = _DIR_GEOM_RE.match(dir_name)
    if not m:
        return {}
    return {
        "gd_wt": float(m.group("gd_wt")),
        "n_gd": int(m.group("n_gd")),
        "axial_zone": f"z{m.group('zone')}",
    }


def _grab_after(tokens: list[str], marker: str) -> float | None:
    for i, tok in enumerate(tokens[:-1]):
        if tok == marker:
            try:
                return float(tokens[i + 1])
            except ValueError:
                return None
    return None


def parse_dec_material(path: str | Path) -> dict:
    """Parse enrichments + Gd content from a ``dec_FA_*.inp`` MATERIAL block.

    Recognises the DeCART ``mixture`` cards:
      * ``UO2``   (92235 x.x)  -> ``enr_main``  (nominal 5.8 w/o)
      * ``UO2_2`` (92235 x.x)  -> ``enr_zone``  (nominal 5.1 w/o, axial zoning)
      * ``UO2G``  (92235 x.x)  -> ``gd_u_enr``  (nominal 4.0 w/o, Gd carrier)
      * the ``6408 x.x`` continuation of ``UO2G`` -> ``gd_wt`` (Gd2O3 wt%)
    """
    out: dict = {}
    in_gd = False
    for raw in _read_text_flex(Path(path)).splitlines():
        toks = raw.split()
        stripped = raw.strip()
        # order matters: UO2_2 / UO2G start with "mixture UO2" too.
        if stripped.startswith("mixture UO2_2"):
            out["enr_zone"] = _grab_after(toks, "92235")
            in_gd = False
        elif stripped.startswith("mixture UO2G"):
            out["gd_u_enr"] = _grab_after(toks, "92235")
            in_gd = True
        elif stripped.startswith("mixture UO2"):
            out["enr_main"] = _grab_after(toks, "92235")
            in_gd = False
        elif in_gd and "6408" in toks:
            out["gd_wt"] = _grab_after(toks, "6408")
            in_gd = False
        elif in_gd and stripped.startswith("mixture"):
            in_gd = False                    # left the UO2G block without a 6408 line
    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------------------- #
# source 3: HGC %DIST Gd-pin heuristic
# --------------------------------------------------------------------------- #
def _first_dist_map_flat(lines: list[str]) -> list[float]:
    """Flatten the first ``%DIST`` 16x16 map (map1 = pin-power form function).

    Shared by :func:`count_gd_pins_from_hgc` (Gd-pin heuristic) and the cond_v4
    ``ff_pin_max`` reader.  ``lines`` may be a whole HGC file or a single %TITL
    block; the *first* ``%DIST`` encountered is used.  Raises :class:`ValueError`
    when no ``%DIST`` block is present.
    """
    di = None
    for i, line in enumerate(lines):
        if line.strip().startswith("%DIST"):
            di = i
            break
    if di is None:
        raise ValueError("no %DIST block")

    rows: list[list[float]] = []
    for line in lines[di + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            break
        try:
            vals = [float(t) for t in stripped.split()]
        except ValueError:
            break
        rows.append(vals)
        if len(rows) >= 16:                  # first BOC 16x16 pin map is enough
            break
    return [v for row in rows for v in row]


def _all_dist_maps_flat(lines: list[str]) -> list[list[float]]:
    """Flatten every stacked 16x16 map of the first ``%DIST`` block.

    A DeCART HGC ``%DIST`` block stacks 7 maps (map1 = pin-power form function ...
    map7 = pin burnup/fluence) as 7*16 = 112 rows of 16 floats.  Returns a list of
    up to 7 flat 256-length maps (fewer if the block is truncated).  Only exact
    16-value rows are accepted, so the parser stops cleanly at the next ``%`` block
    or a short/blank line.  ``[]`` when no ``%DIST`` is present.
    """
    di = None
    for i, line in enumerate(lines):
        if line.strip().startswith("%DIST"):
            di = i
            break
    if di is None:
        return []
    rows: list[list[float]] = []
    for line in lines[di + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            break
        try:
            vals = [float(t) for t in stripped.split()]
        except ValueError:
            break
        if len(vals) != 16:
            break
        rows.append(vals)
        if len(rows) >= 112:                 # 7 stacked 16x16 maps is the whole block
            break
    maps: list[list[float]] = []
    for k in range(len(rows) // 16):
        block = rows[k * 16:(k + 1) * 16]
        maps.append([v for row in block for v in row])
    return maps


def count_gd_pins_from_hgc(path: str | Path) -> int:
    """Count Gd pins from the first (BOC) ``%DIST`` pin-power block of an HGC.

    Gd pins depress the BOC relative pin power well below unity (~0.3-0.4 in the
    reference lattices) while carrying fuel, whereas guide-tube / instrument
    positions read exactly 0.000.  A Gd pin is therefore a position with
    ``0.0 < power < 0.6``.  Validated against the locally-hydrated
    ``0_APR1400/260624/hgc/FA_B1.HGC`` (IGD_20 -> exactly 20).

    Raises :class:`OSError` if the file cannot be read (e.g. an OneDrive
    dehydrated placeholder); callers fall back to the manual anchor.
    """
    with open(path, "r", errors="replace") as handle:
        text = handle.read()
    flat = _first_dist_map_flat(text.splitlines())
    return sum(1 for v in flat if 0.0 < v < 0.6)


# --------------------------------------------------------------------------- #
# cond_v4 physics harvest: reference k-inf curve, BOC reactivity coefficients,
# BOC 2-group cross sections / ADF / pin-power peaking, and the zoning census.
# Shared parser core (.sum and .HGC feed the SAME record shape so a lattice type
# that carries both agrees by construction on kinf0/ff_pin_max — see the cross-
# parity test).  All coefficient sign conventions follow the column contract:
# boron_worth < 0, doppler_coef < 0, cr1_worth > 0, mtc_dmod sign as computed.
# --------------------------------------------------------------------------- #
#: Branch identities (shared by .sum EDIT1 CASE names and HGC 'CASE ::' titles).
_CASE_REF = "REFERENCE CASE"
_CASE_BORON = "BORON VARIATION"
_CASE_TFUEL = "TFUEL VARIATION"
_CASE_DMOD = tuple(f"DMOD{i} VARIATION" for i in range(1, 7))
_CASE_CR1 = "CR1 REFERENCE"

#: Grid of reference burnups at which the k-inf curve is sampled [GWd/tU].
_KINF_GRID = (0.0, 10.0, 20.0, 30.0)

#: Scientific-float token (handles Fortran fixed-format gluing, e.g. a negative
#: buckling ``...E-01-6.29045E-04`` printed with no separating space, and the
#: ``D`` double-exponent form).
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def _floats(line: str) -> list[float]:
    """All scientific floats on a line, robust to glued negative exponentials."""
    out: list[float] = []
    for tok in _FLOAT_RE.findall(line):
        try:
            out.append(float(tok.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    return out


def _rho(k: float) -> float:
    return (k - 1.0) / k


def _interp(xs: list[float], ys: list[float], x: float) -> float | None:
    """Linear interpolation of ``ys`` at ``x`` on ascending ``xs`` (no extrap)."""
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if x <= xs[i]:
            span = xs[i] - xs[i - 1]
            if span == 0.0:
                return ys[i]
            t = (x - xs[i - 1]) / span
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def _burnup_at_k1(xs: list[float], ys: list[float]) -> float | None:
    """First burnup where the (declining) k-inf curve crosses 1.0; None if never."""
    for i in range(1, len(xs)):
        if (ys[i - 1] - 1.0) >= 0.0 > (ys[i] - 1.0):
            drop = ys[i - 1] - ys[i]
            if drop == 0.0:
                return xs[i - 1]
            t = (ys[i - 1] - 1.0) / drop
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None


def _lsq_slope(pts: list[tuple[float, float]]) -> float | None:
    """Least-squares slope dy/dx of ``pts`` (None when x has no spread)."""
    n = len(pts)
    if n < 2:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0.0:
        return None
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return num / den


# --------------------------------------------------------------------------- #
# k-conv (k-inf vs burnup) CURVE-SHAPE features — the poison-agnostic burnable-
# absorber holddown->release signature, read off the SAME reference k-inf curve
# that feeds kinf0..kinf30 (so both the .sum and HGC paths get it for free via
# _curve_and_coeffs, and a type carrying both products agrees by construction).
# --------------------------------------------------------------------------- #
#: Minimum k-inf prominence (hump peak above its preceding trough) for a curve to
#: count as carrying a real burnout hump; below it the curve is treated as
#: near-monotone (weak/absent absorber signature).  ~1e-4 in k is ~8-9 pcm — well
#: above the reference-depletion numerical floor and far below any real hump (the
#: weakest observed swing is hundreds of pcm).
_KCONV_MIN_PROMINENCE: float = 1.0e-4
#: burnup [GWd/tU] at which ``kinf_eol50`` samples the curve.
_KINF_EOL_BU: float = 50.0
#: upper burnup [GWd/tU] of the post-peak quasi-linear depletion-slope fit window.
_KCONV_SLOPE_BU_HI: float = 60.0


def kconv_curve_shape(xs: list[float], ys: list[float]) -> dict:
    """Curve-SHAPE features of a reference k-inf(BU) depletion curve (k-conv).

    ``xs`` are ascending burnups [GWd/tU]; ``ys`` the matching reference k-inf.
    Everything is computed in reactivity space ``rho = (k-1)/k * 1e5`` [pcm] and
    captures the burnable-absorber holddown->release SIGNATURE independently of
    WHICH absorber produced it (Gd today; IFBA/Er/Dy tomorrow) — the shape, not
    the poison design, is the universal feature:

      * ``kinf_dip`` / ``bu_dip_gwd`` — the suppression trough: the minimum k-inf
        on the rising side of the first burnout hump (the local minimum that
        precedes it).  NaN (with ``kconv_is_monotone=1``) for a curve with no
        prominent hump — a weak/absent absorber whose k-inf declines monotonically
        after the initial xenon transient.
      * ``kinf_peak`` / ``bu_peak_gwd`` — the post-dip hump maximum (absorber fully
        burned out).  Degenerates to the BU=0 value for a monotone curve.
      * ``reactivity_swing_pcm`` — ``rho_peak - rho_dip``, the reactivity released
        as the residual absorber between the trough and full burnout depletes (the
        holddown-release magnitude the physics prior consumes directly).  NaN for
        a monotone curve.
      * ``rho_boc_minus_peak_pcm`` — ``rho(0) - rho_peak``: the BOC reactivity
        relative to the hump (dominated by the xenon-free fresh boost; exactly 0
        for a monotone curve, whose peak IS the BU=0 point).
      * ``depletion_slope_pcm_per_gwd`` — least-squares ``d(rho)/dBU`` over the
        post-peak quasi-linear burnout region ``bu_peak .. min(60, last)`` (the
        BU=0 fresh no-xenon point is always excluded); the burnout-region
        reactivity decay rate that drives cycle length.
      * ``kinf_eol50`` — k-inf interpolated at 50 GWd/tU (NaN if the curve stops
        short of it).
      * ``kconv_is_monotone`` — 1.0 when no prominent hump is found, else 0.0.

    Returns only the features it can compute (never raises); ``{}`` for < 3 points.
    """
    if xs is None or ys is None or len(xs) < 3 or len(ys) != len(xs):
        return {}
    n = len(xs)
    kinf0 = ys[0]
    out: dict = {}

    # Locate the burnout hump: the first prominent interior local maximum, with
    # the lowest point up to it as the suppression trough (dip).  The running
    # minimum naturally skips the BU=0 xenon-free spike (a decreasing leg).
    dip_idx = peak_idx = None
    run_min_idx = 0
    for i in range(1, n - 1):
        if ys[i] < ys[run_min_idx]:
            run_min_idx = i
        if (ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]
                and ys[i] - ys[run_min_idx] >= _KCONV_MIN_PROMINENCE):
            peak_idx, dip_idx = i, run_min_idx
            break

    rho0 = _rho(kinf0) * 1e5
    if peak_idx is None:                          # near-monotone: no prominent hump
        out["kconv_is_monotone"] = 1.0
        out["kinf_peak"] = kinf0
        out["bu_peak_gwd"] = xs[0]
        rho_peak = rho0
        peak_bu = xs[0]
    else:
        out["kconv_is_monotone"] = 0.0
        out["kinf_dip"] = ys[dip_idx]
        out["bu_dip_gwd"] = xs[dip_idx]
        out["kinf_peak"] = ys[peak_idx]
        out["bu_peak_gwd"] = xs[peak_idx]
        rho_dip = _rho(ys[dip_idx]) * 1e5
        rho_peak = _rho(ys[peak_idx]) * 1e5
        out["reactivity_swing_pcm"] = rho_peak - rho_dip
        peak_bu = xs[peak_idx]

    out["rho_boc_minus_peak_pcm"] = rho0 - rho_peak

    # post-peak quasi-linear depletion slope (BU=0 fresh point always excluded).
    hi = min(_KCONV_SLOPE_BU_HI, xs[-1])
    seg = [(xs[i], _rho(ys[i]) * 1e5) for i in range(n)
           if xs[i] > 0.0 and peak_bu <= xs[i] <= hi]
    slope = _lsq_slope(seg)
    if slope is not None:
        out["depletion_slope_pcm_per_gwd"] = slope

    eol = _interp(xs, ys, _KINF_EOL_BU)
    if eol is not None:
        out["kinf_eol50"] = eol
    return out


# --------------------------------------------------------------------------- #
# pin-burnup peaking-ratio curve (BRP/BU vs BU) — the physics backbone of the
# serve-side pin-burnup estimator (lpopt.model.pinbu_physics).  Two sources feed
# the SAME (bu, ratio) shape so a type carrying both a .sum and an HGC agrees by
# construction (verified in test_pinbu_physics.test_sum_hgc_pin_bu_parity):
#   * .sum EDIT3     : ratio = BRP / BURNUP at each depletion state.
#   * HGC %DIST map7 : ratio = max(pin-burnup map) / state burnup at each ref state.
# --------------------------------------------------------------------------- #
#: ``ratio(BU) = r_inf + paramA/BU`` is auto-fit over the discharge tail BU >= this
#: (the low-BU states are ~power-peaking and irrelevant to the discharge magnitude).
_PIN_BU_FIT_LO = 25.0
#: minimum in-window points to identify the 2-parameter tail fit.
_PIN_BU_MIN_FIT_PTS = 3


def _pin_bu_points_from_sum(lines: list[str]) -> list[tuple[float, float]]:
    """``(burnup, BRP/burnup)`` points from a ``.sum`` SUMMARY EDIT 3 table.

    EDIT3 row = ``NO BURNUP EFPD FRN FRP BRN BRP ...``; ``BRP`` (peak pin burnup,
    col index 6) over ``BURNUP`` (assembly-average state burnup, col 1) is the
    peak-pin-to-assembly burnup ratio.  Only ``burnup > 0`` states are kept.
    """
    edits = _sum_edit_bounds(lines)
    if 3 not in edits:
        return []
    end3 = edits.get(4, len(lines))
    pts: list[tuple[float, float]] = []
    for ln in lines[edits[3]:end3]:
        toks = ln.split()
        if len(toks) >= 7 and toks[0].isdigit():
            try:
                bu = float(toks[1])
                brp = float(toks[6])
            except ValueError:
                continue
            if bu > 0.0:
                pts.append((bu, brp / bu))
    return pts


def _pin_bu_points_from_hgc(text: str) -> list[tuple[float, float]]:
    """``(burnup, max(map7)/burnup)`` points from an HGC's reference-case states.

    map7 of each ``%DIST`` block is the absolute pin-burnup distribution [GWd/tU];
    its maximum over the state burnup is the peak-pin-to-assembly ratio.  Only the
    reference-case, ``burnup > 0`` blocks contribute (branch cases are skipped).
    """
    pts: list[tuple[float, float]] = []
    for block_lines in _iter_hgc_blocks(text):
        st = _parse_hgc_block_state(block_lines)
        if st is None or st["case"] != _CASE_REF:
            continue
        bu = st["burnup"]
        if bu <= 0.0:
            continue
        maps = _all_dist_maps_flat(block_lines)
        if len(maps) < 7:
            continue
        m7 = maps[6]
        mx = max(m7) if m7 else 0.0
        if mx > 0.0:
            pts.append((bu, mx / bu))
    return pts


def summarize_pin_bu_curve(points: Iterable[tuple[float, float]]) -> dict:
    """Compact summary of a ``(burnup, ratio)`` pin-burnup peaking curve.

    Returns ``pin_bu_ratio_asym`` (ratio at the last/highest state — the discharge
    plateau), ``pin_bu_bu_max`` (that state burnup), ``pin_bu_n_pts`` (point count),
    and the auto-fit ``ratio(BU) = pin_bu_r_inf + pin_bu_paramA/BU`` over the
    discharge tail (``BU >= _PIN_BU_FIT_LO``, needs ``_PIN_BU_MIN_FIT_PTS`` points;
    otherwise the two fit columns are omitted and the estimator leans on the
    plateau).  ``{}`` for an empty curve.
    """
    pts = sorted(
        (float(b), float(r)) for b, r in points
        if b is not None and r is not None
        and math.isfinite(float(b)) and math.isfinite(float(r))
        and float(b) > 0.0 and float(r) > 0.0
    )
    if not pts:
        return {}
    out: dict = {
        "pin_bu_n_pts": float(len(pts)),
        "pin_bu_bu_max": pts[-1][0],
        "pin_bu_ratio_asym": pts[-1][1],
    }
    fit = [(b, r) for b, r in pts if b >= _PIN_BU_FIT_LO]
    if len(fit) >= _PIN_BU_MIN_FIT_PTS:
        xs = [1.0 / b for b, _ in fit]
        ys = [r for _, r in fit]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0.0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            out["pin_bu_paramA"] = slope
            out["pin_bu_r_inf"] = my - slope * mx
    return out


def parse_fa_sum_pin_bu(path: str | Path) -> dict:
    """Standalone pin-burnup curve summary of a ``FA_*.sum`` (EDIT3 BRP column)."""
    lines = _read_text_flex(Path(path)).splitlines()
    return summarize_pin_bu_curve(_pin_bu_points_from_sum(lines))


def parse_hgc_pin_bu(path: str | Path) -> dict:
    """Standalone pin-burnup curve summary of an HGC (%DIST map7 / reference case)."""
    with open(path, "r", errors="replace") as handle:
        text = handle.read()
    return summarize_pin_bu_curve(_pin_bu_points_from_hgc(text))


def _curve_and_coeffs(records: list[dict]) -> dict:
    """Reference k-inf curve points + BOC reactivity coefficients.

    ``records`` is a list of BOC/depletion state dicts, each
    ``{case, burnup, ppm, tfuel, dcool, kinf}`` (``tfuel`` in C or K — only
    differences are used; ``dcool`` in g/cc; ``ppm``).  Both the .sum and .HGC
    parsers build this shape, guaranteeing train-of-thought parity between them.
    """
    by_case: dict[str, list[dict]] = {}
    for r in records:
        by_case.setdefault(r["case"], []).append(r)

    def boc(case: str) -> dict | None:
        rs = by_case.get(case)
        return min(rs, key=lambda r: r["burnup"]) if rs else None

    out: dict = {}
    refs = sorted(by_case.get(_CASE_REF, []), key=lambda r: r["burnup"])
    if refs:
        xs = [r["burnup"] for r in refs]
        ys = [r["kinf"] for r in refs]
        for grid, col in zip(_KINF_GRID, ("kinf0", "kinf10", "kinf20", "kinf30")):
            v = _interp(xs, ys, grid)
            if v is not None:
                out[col] = v
        bk = _burnup_at_k1(xs, ys)
        if bk is not None:
            out["bu_k1"] = bk
        # k-conv CURVE-SHAPE features off the SAME reference curve (poison-agnostic
        # burnable-absorber signature) — .sum and HGC therefore agree by construction.
        out.update(kconv_curve_shape(xs, ys))

    ref = boc(_CASE_REF)
    boron = boc(_CASE_BORON)
    if ref and boron:
        dppm = boron["ppm"] - ref["ppm"]
        if dppm:
            out["boron_worth"] = (_rho(boron["kinf"]) - _rho(ref["kinf"])) / dppm * 1e5
    tfuel = boc(_CASE_TFUEL)
    if ref and tfuel:
        dtf = tfuel["tfuel"] - ref["tfuel"]
        if dtf:
            out["doppler_coef"] = (_rho(tfuel["kinf"]) - _rho(ref["kinf"])) / dtf * 1e5
    dmod_pts = [
        (d["dcool"], _rho(d["kinf"]) * 1e5)
        for d in (boc(c) for c in _CASE_DMOD) if d
    ]
    slope = _lsq_slope(dmod_pts)
    if slope is not None:
        out["mtc_dmod"] = slope * 0.01               # per 0.01 g/cc
    cr1 = boc(_CASE_CR1)
    if ref and cr1:
        out["cr1_worth"] = (_rho(ref["kinf"]) - _rho(cr1["kinf"])) * 1e5
    return out


# --------------------------------------------------------------------------- #
# .sum parser (SUMMARY EDIT 1/2/3)
# --------------------------------------------------------------------------- #
_EDIT_RE = re.compile(r"\s*SUMMARY EDIT (\d)")


def _sum_edit_bounds(lines: list[str]) -> dict[int, int]:
    edits: dict[int, int] = {}
    for i, ln in enumerate(lines):
        m = _EDIT_RE.match(ln)
        if m:
            edits[int(m.group(1))] = i
    return edits


def parse_fa_sum(path: str | Path) -> dict:
    """Harvest the cond_v4 physics of a DeCART ``FA_*.sum`` file.

    Returns ``kinf0/kinf10/kinf20/kinf30`` (reference-depletion k-inf sampled on
    ``_KINF_GRID``), ``bu_k1`` (root of k-inf = 1 on that curve), ``ff_pin_max``
    (EDIT3 ``FRP`` at BU=0), and the BOC reactivity coefficients
    ``boron_worth / doppler_coef / mtc_dmod / cr1_worth``.  Branch identities come
    from the EDIT1 ``CASE`` names; each coefficient uses the BURNUP=0 row of its
    branch sweep, joined to EDIT2 by row number.  Missing pieces are simply
    omitted (never raise) so a truncated file degrades gracefully.

    Does not cover ``xs_*`` / ``adf_*`` (HGC-only) or ``zone_pin_count`` (inp).
    """
    lines = _read_text_flex(Path(path)).splitlines()
    edits = _sum_edit_bounds(lines)
    if 1 not in edits or 2 not in edits:
        return {}
    end2 = edits.get(3, len(lines))

    # EDIT1: row number -> CASE label (label may contain spaces; the trailing
    # three columns are BURNUP EFPD XE, so split from the right).
    case_by_no: dict[int, str] = {}
    for ln in lines[edits[1]:edits[2]]:
        toks = ln.split()
        if len(toks) >= 5 and toks[0].isdigit():
            case_by_no[int(toks[0])] = " ".join(toks[1:-3])

    # EDIT2 REACTIVITY: NO BURNUP EFPD PPM TF TCOO DCOO PGAM K-CONV K-CRIT B-SQ.
    records: list[dict] = []
    for ln in lines[edits[2]:end2]:
        toks = ln.split()
        if len(toks) < 11 or not toks[0].isdigit():
            continue
        case = case_by_no.get(int(toks[0]))
        if case is None:
            continue
        try:
            records.append(dict(
                case=case,
                burnup=float(toks[1]),
                ppm=float(toks[3]),
                tfuel=float(toks[4]),
                dcool=float(toks[6]),
                kinf=float(toks[8]),             # K-CONV = k-inf
            ))
        except ValueError:
            continue
    out = _curve_and_coeffs(records)

    # EDIT3: reference BU=0 FRP (pin radial power peaking) = NO BU EFPD FRN FRP...
    if 3 in edits:
        end3 = edits.get(4, len(lines))
        for ln in lines[edits[3]:end3]:
            toks = ln.split()
            if len(toks) >= 5 and toks[0] == "1":
                try:
                    out["ff_pin_max"] = float(toks[4])
                except ValueError:
                    pass
                break
        # pin-burnup peaking-ratio curve (BRP/BU vs BU) from the SAME EDIT3 table.
        out.update(summarize_pin_bu_curve(_pin_bu_points_from_sum(lines)))
    return out


# --------------------------------------------------------------------------- #
# HGC parser (%TITL state scalars, %MACX, %ADFT, %DIST)
# --------------------------------------------------------------------------- #
def _parse_hgc_block_state(block_lines: list[str]) -> dict | None:
    """CASE label + BOC state scalars from one %TITL block.

    Layout after the ``CASE ::`` line: an integer descriptor line, then
    ``lineA = sp_power BURNUP kinf kcrit Bsq Tfuel`` and
    ``lineB = Tcool ppm _ pressure rho_cool pnorm``.
    """
    case = None
    cidx = None
    for i, ln in enumerate(block_lines):
        if "CASE ::" in ln:
            case = ln.split("CASE ::", 1)[1].strip()
            cidx = i
            break
    if case is None:
        return None
    nums: list[list[float]] = []
    for ln in block_lines[cidx + 1:cidx + 7]:
        vals = _floats(ln)
        if vals:
            nums.append(vals)
        if len(nums) >= 3:
            break
    if len(nums) < 3:
        return None
    line_a, line_b = nums[1], nums[2]
    if len(line_a) < 6 or len(line_b) < 6:
        return None
    return dict(
        case=case,
        burnup=line_a[1],
        kinf=line_a[2],
        tfuel=line_a[5],
        tcool=line_b[0],
        ppm=line_b[1],
        dcool=line_b[4],
    )


def _parse_hgc_macx_adf(block_lines: list[str]) -> dict:
    """BOC 2-group ``xs_*`` (%MACX) and ``adf_*`` (%ADFT) from one %TITL block."""
    out: dict = {}
    for i, ln in enumerate(block_lines):
        s = ln.strip()
        if s.startswith("%MACX"):
            g1 = _floats(block_lines[i + 1])
            g2 = _floats(block_lines[i + 2])
            scat1 = _floats(block_lines[i + 3])     # 1->1, 1->2
            if len(g1) >= 6 and len(g2) >= 6:
                out.update(
                    xs_d1=g1[0], xs_a1=g1[1], xs_nf1=g1[3],
                    xs_d2=g2[0], xs_a2=g2[1], xs_nf2=g2[3],
                )
                # s12 = scatter 1->2: matrix row-1 col-2, == group-1 out-scatter.
                if len(scat1) >= 2:
                    out["xs_s12"] = scat1[1]
                else:
                    out["xs_s12"] = g1[5]
            break
    for i, ln in enumerate(block_lines):
        if ln.strip().startswith("%ADFT"):
            f_g1 = _floats(block_lines[i + 1])
            f_g2 = _floats(block_lines[i + 2])
            c_g1 = _floats(block_lines[i + 3])
            c_g2 = _floats(block_lines[i + 4])
            if f_g1 and f_g2 and c_g1 and c_g2:
                out.update(
                    adf_face_g1=f_g1[0], adf_face_g2=f_g2[0],
                    adf_corner_g1=c_g1[0], adf_corner_g2=c_g2[0],
                )
            break
    return out


def _iter_hgc_blocks(text: str) -> Iterable[list[str]]:
    for blk in re.split(r"(?m)^%TITL", text):
        if blk.strip():
            yield blk.splitlines()


def parse_hgc_full(path: str | Path) -> dict:
    """Harvest the cond_v4 physics of an HGC file (every column except census).

    Mirrors :func:`parse_fa_sum` for the reference k-inf curve + BOC reactivity
    coefficients (identical record shape -> byte-level agreement on shared types)
    and additionally sources the HGC-only ``xs_*`` (%MACX), ``adf_*`` (%ADFT),
    and ``ff_pin_max`` (%DIST map1 max) from the first reference-BOC block.

    Raises :class:`OSError` if the file cannot be read (OneDrive dehydrated
    placeholder); callers fall back to the manual anchor.  All parse gaps are
    silently omitted from the returned dict.
    """
    with open(path, "r", errors="replace") as handle:
        text = handle.read()

    records: list[dict] = []
    boc_extra: dict = {}
    for block_lines in _iter_hgc_blocks(text):
        st = _parse_hgc_block_state(block_lines)
        if st is None:
            continue
        records.append(st)
        if (not boc_extra and st["case"] == _CASE_REF
                and abs(st["burnup"]) < 1.0e-6):
            boc_extra = _parse_hgc_macx_adf(block_lines)
            try:
                flat = _first_dist_map_flat(block_lines)
                if flat:
                    boc_extra["ff_pin_max"] = max(flat)
            except ValueError:
                pass
    out = _curve_and_coeffs(records)
    out.update(boc_extra)
    # pin-burnup peaking-ratio curve (%DIST map7 across the reference case) — the
    # HGC-only fallback for ga80 / a paramA package that stages no sibling .sum.
    out.update(summarize_pin_bu_curve(_pin_bu_points_from_hgc(text)))
    return out


def parse_hgc_boc_xs_adf(path: str | Path) -> dict:
    """HGC-only ``xs_*`` / ``adf_*`` / ``ff_pin_max`` from the first BOC block.

    Light-weight companion to :func:`parse_hgc_full` used when the reference
    curve + coefficients already come from a sibling ``.sum`` (lattice dirs):
    stops at the first reference-BOC block instead of scanning all 334 blocks.
    """
    with open(path, "r", errors="replace") as handle:
        text = handle.read()
    for block_lines in _iter_hgc_blocks(text):
        st = _parse_hgc_block_state(block_lines)
        if st is None:
            continue
        if st["case"] == _CASE_REF and abs(st["burnup"]) < 1.0e-6:
            out = _parse_hgc_macx_adf(block_lines)
            try:
                flat = _first_dist_map_flat(block_lines)
                if flat:
                    out["ff_pin_max"] = max(flat)
            except ValueError:
                pass
            return out
    return {}


# --------------------------------------------------------------------------- #
# dec_FA_*.inp octant census -> zone_pin_count (+ n_gd cross-check)
# --------------------------------------------------------------------------- #
_CELL_RE = re.compile(r"^cell\s+(\d+)\s+\d+\s*/\s*(\S+)")
_ASSEMBLY_RE = re.compile(r"^assembly\s+\S+\s+45\s+1\b")


def parse_dec_zone_census(path: str | Path) -> dict:
    """Zoning-pin (and Gd-pin cross-check) census from a ``dec_FA_*.inp`` octant.

    The octant map after ``assembly <name> 45 1`` is eight triangular rows of
    cell-IDs (lengths 1..8, 36 positions).  Position ``(i, j)`` (0-indexed) has
    multiplicity 4 on the diagonal (``j == i``) and 8 otherwise, summing to the
    full 16x16 = 256 pins.  Cell-IDs whose first material is ``UO2_2`` are zoning
    pins (``zone_pin_count``); ``UO2G`` are Gd pins (``n_gd_census``, used only to
    cross-check the directory ``n_gd``).  Returns ``{}`` when no census is found.
    """
    lines = _read_text_flex(Path(path)).splitlines()

    cell_mat: dict[int, str] = {}
    for ln in lines:
        m = _CELL_RE.match(ln.strip())
        if m:
            cell_mat[int(m.group(1))] = m.group(2)
    zone_cells = {c for c, mat in cell_mat.items() if mat == "UO2_2"}
    gd_cells = {c for c, mat in cell_mat.items() if mat == "UO2G"}

    ai = None
    for i, ln in enumerate(lines):
        if _ASSEMBLY_RE.match(ln.strip()):
            ai = i
            break
    if ai is None:
        return {}

    rows: list[list[int]] = []
    expected = 1
    for ln in lines[ai + 1:]:
        s = ln.strip()
        if not s:
            if rows:
                break
            continue
        try:
            ints = [int(t) for t in s.split()]
        except ValueError:
            break
        if len(ints) != expected:
            break
        rows.append(ints)
        expected += 1
        if len(rows) >= 8:
            break
    if not rows:
        return {}

    zone = 0
    gd = 0
    for i, row in enumerate(rows):
        for j, cid in enumerate(row):
            mult = 4 if j == i else 8
            if cid in zone_cells:
                zone += mult
            if cid in gd_cells:
                gd += mult
    return {"zone_pin_count": zone, "n_gd_census": gd}


# --------------------------------------------------------------------------- #
# dec_FA_*.inp GEOM block -> pin-cell geometry (pitch + fuel-pin radii)
# --------------------------------------------------------------------------- #
#: nominal assembly pitch [cm] — the frozen envelope anchor (== coredeck
#: CoreParams.wide).  Harvested ``asm_pitch`` is asserted against this by the
#: geometry-validation deck editor so a variant can never move the MASTER core mesh.
NOMINAL_ASM_PITCH: float = 20.7772


def geom_derived(pin_pitch: float, r_pellet: float, r_clad_out: float) -> dict:
    """Derived single-cell moderation ratios from pin pitch + fuel-pin radii.

    ``p_over_d = pin_pitch / (2*r_clad_out)`` (pitch-to-diameter);
    ``v_mod_over_v_fuel = (pitch^2 - pi*r_clad_out^2) / (pi*r_pellet^2)`` — the
    single-cell moderator-to-fuel volume ratio (guide tubes ignored), nominal 1.788.
    Returns ``{}`` when an input is non-positive.
    """
    if not (pin_pitch > 0.0 and r_pellet > 0.0 and r_clad_out > 0.0):
        return {}
    fuel_area = math.pi * r_pellet * r_pellet
    mod_area = pin_pitch * pin_pitch - math.pi * r_clad_out * r_clad_out
    out = {"p_over_d": pin_pitch / (2.0 * r_clad_out)}
    if fuel_area > 0.0:
        out["v_mod_over_v_fuel"] = mod_area / fuel_area
    return out


def parse_dec_geom(path: str | Path) -> dict:
    """Parse the pin-cell geometry from a ``dec_FA_*.inp`` GEOM block (review 4a).

    Reads the ``pitch`` card (``pitch <pin_pitch> <asm_pitch>``) and the normal
    fuel-pin ``cellgeo 1`` card (``cellgeo 1 <r_pellet> <r_clad_in> <r_clad_out>
    / ...``); ``npins`` is parsed only as a QC cross-check.  Returns the harvested
    ``pin_pitch / asm_pitch / r_pellet / r_clad_in / r_clad_out`` plus the derived
    ``p_over_d`` / ``v_mod_over_v_fuel``.  Missing pieces are simply omitted (never
    raises) so a truncated / non-lattice deck degrades to ``{}``.
    """
    out: dict = {}
    r_pellet = r_clad_out = None
    for raw in _read_text_flex(Path(path)).splitlines():
        toks = raw.split()
        if not toks:
            continue
        head = toks[0].lower()
        if head == "pitch" and len(toks) >= 3:
            try:
                out["pin_pitch"] = float(toks[1])
                out["asm_pitch"] = float(toks[2])
            except ValueError:
                pass
        elif head == "cellgeo" and len(toks) >= 3 and toks[1] == "1":
            # radii are the floats between "cellgeo 1" and the first "/".
            radii: list[float] = []
            for t in toks[2:]:
                if t == "/":
                    break
                try:
                    radii.append(float(t))
                except ValueError:
                    break
            if len(radii) >= 3:
                out["r_pellet"] = r_pellet = radii[0]
                out["r_clad_in"] = radii[1]
                out["r_clad_out"] = r_clad_out = radii[2]
    if out.get("pin_pitch") and r_pellet and r_clad_out:
        out.update(geom_derived(out["pin_pitch"], r_pellet, r_clad_out))
    return out


# --------------------------------------------------------------------------- #
# source 4: legacy A0/A1  (MOCHA config.py L78-81, D:\eqlp_ws cache reality)
# --------------------------------------------------------------------------- #
def legacy_rows() -> list[FuelVec]:
    """Hard-coded legacy A0/A1 designs (library ``legacy_a``)."""
    return [
        FuelVec(
            library_id="legacy_a", type_id="A0",
            u_avg_enrichment=5.6345, enr_main=5.8, enr_zone=5.1, n_gd=16,
            source_flags=["mocha_hardcoded"], feature_poor=False,
        ),  # 5.8 x168 + 5.1 x52, 16 Gd
        FuelVec(
            library_id="legacy_a", type_id="A1",
            u_avg_enrichment=5.6375, enr_main=5.8, enr_zone=5.1, n_gd=12,
            source_flags=["mocha_hardcoded"], feature_poor=False,
        ),  # 5.8 x172 + 5.1 x52, 12 Gd
    ]


# --------------------------------------------------------------------------- #
# parsed lattice libraries (260624 / 5.8_5.1 / CPHA)
# --------------------------------------------------------------------------- #
# The 5.8_5.1 IGD_16 lattices are stored on disk as A01..A06 but are byte-
# identical HGC renames of the X-series slots X2,X3,X6,X7,X10,X11 (plan 4.3 /
# section 1-6 "5.8_5.1 X 12종").  Both id rows are emitted, cross-referenced in
# source_flags.  (Verified against the X-index arithmetic
# 4*{6:0,8:1,10:2}[gd_wt] + 2*{12:0,16:1}[n_gd] + {z1:0,z2:1}[zone].)
_LIB_5851_ALIAS: dict[str, str] = {
    "A01": "X2", "A02": "X3", "A03": "X6", "A04": "X7", "A05": "X10", "A06": "X11",
}

#: library_id -> (FA subtree relative to apr1400_root, alias map)
_PARSED_LIBS: dict[str, tuple[str, dict[str, str]]] = {
    "260624": ("260624/FA", {}),
    "5.8_5.1": ("5.8_5.1/FA", _LIB_5851_ALIAS),
    "CPHA": ("CPHA/FA", {}),
}


def _natural_key(text: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


# --------------------------------------------------------------------------- #
# cond_v4 harvest wiring (fill a FuelVec from parsed .sum/.HGC/inp dicts)
# --------------------------------------------------------------------------- #
#: FuelVec float attributes fillable by the physics harvest (curve + cond_v4).
_HARVEST_FIELDS: tuple[str, ...] = (
    "kinf0", "kinf10", "kinf20", "kinf30", "bu_k1", *COND_V4_COLUMNS,
    *PIN_BU_COLUMNS, *GEOM_COLUMNS, *KCONV_SHAPE_COLUMNS,
)


def _apply_harvest(vec: FuelVec, data: dict) -> None:
    """Copy finite harvest values onto ``vec`` (skip None / NaN)."""
    for name in _HARVEST_FIELDS:
        v = data.get(name)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        setattr(vec, name, fv)


def harvest_lattice_cond_v4(out_path: Path) -> tuple[dict, list[str]]:
    """Harvest the cond_v4 physics for a lattice type from files beside ``out_path``.

    Precedence per the column contract: HGC supplies ``xs_*`` / ``adf_*`` (and a
    fallback ``ff_pin_max``); the sibling ``.sum`` overrides the k-inf curve,
    ``bu_k1``, ``ff_pin_max`` (FRP), and the BOC reactivity coefficients; the
    ``dec_FA_*.inp`` octant supplies ``zone_pin_count``.  Returns
    ``(harvest_dict, source_flags)``; every source is best-effort (a missing or
    dehydrated file just leaves those columns NaN).
    """
    data: dict = {}
    flags: list[str] = []

    sum_path = out_path.with_suffix(".sum")
    have_sum = sum_path.is_file()

    # The HGC is usually ``FA_<type>.HGC`` but some libraries (5.8_5.1 B-series)
    # emit a state-point-suffixed ``FA_<type>_0101.HGC`` — accept either.
    stem = out_path.name[: -len(out_path.suffix)] if out_path.suffix else out_path.name
    hgc_path = out_path.with_suffix(".HGC")
    if not hgc_path.is_file():
        variants = sorted(out_path.parent.glob(f"{stem}_*.HGC"))
        hgc_path = variants[0] if variants else hgc_path

    # HGC: when a sibling .sum will supply the curve + coefficients we only need
    # the HGC-only xs/adf (+ ff fallback) from the first block; otherwise (e.g.
    # a paramA package that stages no .sum) parse the whole HGC for everything.
    if hgc_path.is_file():
        try:
            if have_sum:
                hgc = parse_hgc_boc_xs_adf(hgc_path)
                hgc_flag = "hgc_xs"
            else:
                hgc = parse_hgc_full(hgc_path)
                hgc_flag = "hgc_full"
        except OSError:
            hgc, hgc_flag = {}, None
        if hgc:
            data.update(hgc)
            if hgc_flag:
                flags.append(hgc_flag)

    if have_sum:
        try:
            s = parse_fa_sum(sum_path)
        except (OSError, ValueError):
            s = {}
        if s:
            data.update(s)                       # .sum wins for kinf/FRP/branches
            flags.append("fa_sum")

    for dec in sorted(out_path.parent.glob("dec_FA_*.inp")):
        try:
            cen = parse_dec_zone_census(dec)
        except (OSError, ValueError):
            cen = {}
        if "zone_pin_count" in cen:
            data["zone_pin_count"] = float(cen["zone_pin_count"])
            flags.append("dec_census")
        # pin-cell geometry (pitch + fuel-pin radii) from the SAME dec inp.
        try:
            geom = parse_dec_geom(dec)
        except (OSError, ValueError):
            geom = {}
        if geom:
            data.update(geom)
            flags.append("dec_geom")
        break
    return data, flags


def rows_from_fa_dir(library_id: str, fa_root: Path,
                     alias_map: dict[str, str] | None = None) -> list[FuelVec]:
    """Build FuelVecs from every ``FA_*.out`` under ``fa_root`` (source 1+2).

    Each row combines the FA MASS(g) inventory, the sibling ``dec_FA_*.inp``
    MATERIAL block, and the parent directory name geometry.
    """
    alias_map = alias_map or {}
    rows: list[FuelVec] = []
    if not fa_root.is_dir():
        return rows

    seen: dict[str, Path] = {}
    for out_path in sorted(fa_root.rglob("*.out"), key=lambda p: _natural_key(p.name)):
        m = _FA_OUT_RE.match(out_path.name)
        if not m:
            continue
        type_id = m.group("name")
        if type_id in seen:
            raise ValueError(
                f"duplicate FA OUT for {library_id}/{type_id}: "
                f"{seen[type_id]} and {out_path}"
            )
        seen[type_id] = out_path

        mass = parse_fa_mass_out(out_path)
        flags = ["fa_mass_out"]

        geom: dict = parse_dir_geometry(out_path.parent.name)
        if geom:
            flags.append("dir_geom")

        material: dict = {}
        dec_files = sorted(out_path.parent.glob("dec_FA_*.inp"))
        if dec_files:
            material = parse_dec_material(dec_files[0])
            if material:
                flags.append("dec_material")

        vec = FuelVec(
            library_id=library_id,
            type_id=type_id,
            u_avg_enrichment=mass["u_avg_enrichment"],
            u_mass_g=mass["u_mass_g"],
            enr_main=material.get("enr_main"),
            enr_zone=material.get("enr_zone"),
            gd_u_enr=material.get("gd_u_enr"),
            # directory name is authoritative for gd_wt / n_gd; MATERIAL 6408 is
            # a fallback for gd_wt only.
            gd_wt=geom.get("gd_wt", material.get("gd_wt")),
            n_gd=geom.get("n_gd"),
            axial_zone=geom.get("axial_zone"),
            source_flags=flags,
            feature_poor=False,
        )

        # cond_v4 physics: reference k-inf curve + BOC coefficients (.sum),
        # 2-group xs / ADF (HGC), zoning census (dec inp).  Filled onto the vec
        # (and inherited by any alias copy below).
        cond, cond_flags = harvest_lattice_cond_v4(out_path)
        _apply_harvest(vec, cond)
        # cond_v4 provenance flags are appended AFTER the existing (and alias)
        # flags so the pre-cond_v4 flag list stays a prefix of the new one.
        alias = alias_map.get(type_id)
        if alias is not None:
            vec.source_flags = flags + [f"alias:{alias}"] + cond_flags
            alias_vec = FuelVec(**{**vec.__dict__})
            alias_vec.type_id = alias
            alias_vec.source_flags = flags + [f"alias_of:{type_id}"] + cond_flags
            rows.append(vec)
            rows.append(alias_vec)
        else:
            vec.source_flags = flags + cond_flags
            rows.append(vec)
    return rows


# --------------------------------------------------------------------------- #
# source 5: manual ga80 anchors + source 3 HGC Gd counts
# --------------------------------------------------------------------------- #
#: ga80 letter roster (sel32_pairs.py: 32 base ABCD x8 + codesign E/F/M +
#: codesign2 G + codesign3 H/J/K/L/N).  Family -> number of designs.
GA80_FAMILIES: dict[str, int] = {
    "A": 8, "B": 8, "C": 8, "D": 8, "E": 4, "F": 6, "G": 4,
    "H": 6, "J": 6, "K": 6, "L": 6, "M": 4, "N": 6,
}
GA80_TYPE_IDS: list[str] = [
    f"{fam}{i}" for fam, n in GA80_FAMILIES.items() for i in range(1, n + 1)
]


def load_manual_anchors(manual_yaml: str | Path) -> tuple[dict[str, float], set[str]]:
    """Load ``(anchors, excluded_families)`` from ``fuel_types_manual.yaml``."""
    doc = yaml.safe_load(_read_text_flex(Path(manual_yaml))) or {}
    anchors = {str(k): float(v) for k, v in (doc.get("anchors") or {}).items()}
    excluded = {str(x) for x in (doc.get("excluded") or [])}
    return anchors, excluded


def _hgc_gd_counts(hgc_dir: Path) -> tuple[dict[str, int], list[str]]:
    """Return ``({type_id: n_gd}, dehydrated_type_ids)`` for readable HGCs.

    Every ``FA_*.HGC`` under ``hgc_dir`` is attempted individually; an unreadable
    file (OneDrive dehydrated placeholder, typically ``[Errno 22]``) is recorded
    as dehydrated and skipped so the type falls through to the manual anchor.
    """
    counts: dict[str, int] = {}
    dehydrated: list[str] = []
    if not hgc_dir.is_dir():
        return counts, dehydrated
    for p in sorted(hgc_dir.glob("FA_*.HGC")):
        type_id = p.name[len("FA_"):-len(".HGC")]
        try:
            counts[type_id] = count_gd_pins_from_hgc(p)
        except OSError:
            dehydrated.append(type_id)
    return counts, dehydrated


def _hgc_cond_map(hgc_dir: Path) -> dict[str, dict]:
    """``{type_id: cond_v4 harvest dict}`` for readable ga80 HGCs (HGC-only path).

    Each ``FA_*.HGC`` yields the full k-inf curve + BOC reactivity coefficients +
    2-group xs / ADF / pin-power peaking via :func:`parse_hgc_full`.  Dehydrated
    placeholders (``OSError``) are skipped so the type keeps its manual anchor.
    """
    out: dict[str, dict] = {}
    if not hgc_dir.is_dir():
        return out
    for p in sorted(hgc_dir.glob("FA_*.HGC")):
        type_id = p.name[len("FA_"):-len(".HGC")]
        try:
            data = parse_hgc_full(p)
        except OSError:
            continue
        if data:
            out[type_id] = data
    return out


def ga80_rows(manual_yaml: str | Path, hgc_dir: Path
              ) -> tuple[list[FuelVec], set[str], dict[str, int], list[str]]:
    """Build the ga80 letter library from HGC Gd counts + manual anchors.

    Returns ``(rows, excluded_type_ids, hgc_gd_counts, dehydrated_type_ids)``.
    A type whose family is excluded (F/M) emits no row.  A type with a readable
    HGC gets ``n_gd`` from source 3 and ``feature_poor=False``; otherwise it
    carries the manual anchor alone with ``feature_poor=True``.
    """
    anchors, excluded_families = load_manual_anchors(manual_yaml)
    hgc_gd, dehydrated = _hgc_gd_counts(hgc_dir)
    hgc_cond = _hgc_cond_map(hgc_dir)

    rows: list[FuelVec] = []
    excluded_ids: set[str] = set()
    for type_id in GA80_TYPE_IDS:
        family = type_id[0]
        if family in excluded_families:
            excluded_ids.add(type_id)
            continue
        flags = ["manual_anchor"]
        n_gd = hgc_gd.get(type_id)
        if n_gd is not None:
            flags = ["hgc_gd_dist", "manual_anchor"]
        vec = FuelVec(
            library_id="ga80",
            type_id=type_id,
            u_avg_enrichment=anchors.get(family),
            n_gd=n_gd,
            source_flags=flags,
            feature_poor=(n_gd is None),     # Gd design unknown -> feature-poor
        )
        # cond_v4 physics from the same HGC (kinf curve + coefficients + xs/adf +
        # ff_pin_max).  zone_pin_count stays NaN — ga80 ships no dec inp.
        cond = hgc_cond.get(type_id)
        if cond:
            _apply_harvest(vec, cond)
            vec.source_flags = vec.source_flags + ["hgc_full"]
        rows.append(vec)
    return rows, excluded_ids, hgc_gd, dehydrated


# --------------------------------------------------------------------------- #
# paramA parametric-design library (plan section 12)
# --------------------------------------------------------------------------- #
def paramA_rows(paramA_root: str | Path) -> list[FuelVec]:
    """Build the ``paramA`` fuel rows from a design directory (plan 12.2).

    The directory carries ``designs.json`` — a list of design records
    ``{type_id, alias, e1, e2, zoning_variant, gd_wt, n_gd}`` — and the
    ``FA_<alias>.out`` DeCART inventories (searched recursively).  Every row is
    fully featured (``feature_poor=False``): the MASS(g) inventory supplies
    ``u_avg_enrichment`` / ``u_mass_g`` and the design axes supply the rest.  A
    design whose ``FA_<alias>.out`` is missing is skipped (not yet harvested).
    """
    root = Path(paramA_root)
    manifest = root / "designs.json"
    if not manifest.is_file():
        return []
    doc = json.loads(_read_text_flex(manifest))
    designs = doc.get("designs", doc) if isinstance(doc, dict) else doc

    out_index: dict[str, Path] = {}
    for p in root.rglob("FA_*.out"):
        m = _FA_OUT_RE.match(p.name)
        if m:
            out_index.setdefault(m.group("name"), p)

    # HGC index (by alias) so a package that staged only ``FA_<alias>.HGC`` (no
    # ``.out``) still auto-harvests the cond_v4 physics from its HGC.
    hgc_index: dict[str, Path] = {}
    for p in root.rglob("FA_*.HGC"):
        stem = p.name[len("FA_"):-len(".HGC")]
        alias_key = stem.split("_")[0]           # FA_<alias>[_<state>].HGC
        hgc_index.setdefault(alias_key, p)
        hgc_index.setdefault(stem, p)

    rows: list[FuelVec] = []
    for d in designs:
        alias = str(d["alias"])
        type_id = str(d.get("type_id", alias))
        out_path = out_index.get(alias)
        flags = ["paramA_design", f"alias:{alias}", f"design:{type_id}"]
        u_enr = None
        u_mass = None
        if out_path is not None:
            try:
                mass = parse_fa_mass_out(out_path)
                u_enr = mass["u_avg_enrichment"]
                u_mass = mass["u_mass_g"]
                flags.append("fa_mass_out")
            except (OSError, ValueError):
                pass
        vec = FuelVec(
            library_id="paramA",
            type_id=type_id,
            u_avg_enrichment=u_enr,
            enr_main=float(d["e1"]),
            enr_zone=float(d["e2"]),
            u_mass_g=u_mass,
            n_gd=int(d["n_gd"]),
            gd_wt=float(d["gd_wt"]),
            gd_u_enr=float(d.get("gd_u_enr", 4.0)),
            axial_zone=str(d["zoning_variant"]),
            source_flags=flags,
            feature_poor=False,
        )
        # cond_v4 physics from the staged FA_<alias>.sum / .HGC (+ dec inp if any).
        # No .sum in a package -> parse_hgc_full covers the whole curve + coeffs.
        # Anchor sibling-derivation on the .out when present, else on the HGC.
        anchor = out_path
        if anchor is None:
            hgc = hgc_index.get(alias)
            if hgc is not None:
                anchor = hgc.parent / f"FA_{alias}.out"
        if anchor is not None:
            cond, cond_flags = harvest_lattice_cond_v4(anchor)
            _apply_harvest(vec, cond)
            vec.source_flags = flags + cond_flags
        rows.append(vec)
    return rows


# --------------------------------------------------------------------------- #
# build + persist
# --------------------------------------------------------------------------- #
def _rows_to_frame(rows: Iterable[FuelVec]) -> pd.DataFrame:
    rows = list(rows)
    records = [r.to_record() for r in rows]
    df = pd.DataFrame.from_records(records, columns=SCHEMA_COLUMNS)
    for col in _FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["n_gd"] = pd.array(
        [None if r.n_gd is None else int(r.n_gd) for r in rows], dtype="Int64"
    )
    df["feature_poor"] = df["feature_poor"].astype(bool)
    return df


def build_fuel_table(cfg_or_paths: Any, *, persist: bool = True) -> pd.DataFrame:
    """Assemble the ``(library_id, type_id)`` fuel-feature table (plan 4.3).

    ``cfg_or_paths`` may be an :class:`lpopt.config.LpoptConfig`, a
    :class:`FuelPaths`, or a ``dict`` of the four paths.  When ``persist`` is
    true the table is written to ``paths.store`` (``data/store/fuel_types.parquet``
    by default).
    """
    paths = _as_fuel_paths(cfg_or_paths)

    rows: list[FuelVec] = []
    for library_id, (fa_rel, alias) in _PARSED_LIBS.items():
        rows.extend(rows_from_fa_dir(library_id, paths.apr1400_root / fa_rel, alias))
    rows.extend(legacy_rows())
    ga_rows, _excluded, _hgc_gd, _dehydrated = ga80_rows(paths.manual_yaml, paths.ga80_hgc)
    rows.extend(ga_rows)
    if paths.paramA_root is not None:
        rows.extend(paramA_rows(paths.paramA_root))

    # (library_id, type_id) is the key — duplicates are a build bug.
    keys = [r.key for r in rows]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate (library_id, type_id) rows: {sorted(dupes)}")

    df = _rows_to_frame(rows)
    if persist:
        paths.store.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(paths.store, engine="pyarrow", index=False)
    return df


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Write ``df`` to ``path`` atomically (tmp + replace) so a live reader of the
    store never sees a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)


def augment_fuel_table_pin_bu(cfg_or_paths: Any, *, persist: bool = True
                              ) -> pd.DataFrame:
    """Additively add the :data:`PIN_BU_COLUMNS` to an existing fuel-types parquet.

    A full :func:`build_fuel_table` only emits the libraries whose sources are
    resolvable now (e.g. it drops ``paramA`` rows when ``design.paramA_root`` is
    unset), so it is NOT a safe rebuild of a store that already carries such rows.
    This helper instead **preserves every existing row and column byte-for-byte**
    and only appends the pin-burnup curve columns: it harvests the curve for the
    resolvable ``(library_id, type_id)`` keys (via :func:`build_fuel_table` in
    memory) and left-joins them onto the on-disk store; any key with no harvestable
    source (paramA/legacy/dehydrated ga80) keeps ``NaN`` pin-burnup columns.

    Idempotent: re-running drops and re-derives the pin-burnup columns.  When
    ``persist`` the result is written atomically back to ``paths.store``.
    """
    paths = _as_fuel_paths(cfg_or_paths)
    existing = pd.read_parquet(paths.store)
    fresh = build_fuel_table(paths, persist=False)

    cols = list(PIN_BU_COLUMNS)
    key = ["library_id", "type_id"]
    pin = fresh[key + cols].drop_duplicates(subset=key)
    base = existing.drop(columns=[c for c in cols if c in existing.columns])
    merged = base.merge(pin, on=key, how="left")           # left-join preserves order
    for c in cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")
    # keep the persisted column order deterministic (existing order + new tail).
    ordered = [c for c in SCHEMA_COLUMNS if c in merged.columns]
    ordered += [c for c in merged.columns if c not in ordered]
    merged = merged[ordered]

    if len(merged) != len(existing):
        raise ValueError(
            f"pin-burnup augment changed row count {len(existing)} -> {len(merged)}"
        )
    if persist:
        _atomic_write_parquet(paths.store, merged)
    return merged


def augment_fuel_table_geometry(cfg_or_paths: Any, *, persist: bool = True
                                ) -> pd.DataFrame:
    """Additively add the :data:`GEOM_COLUMNS` to an existing fuel-types parquet.

    Same preserve-everything contract as :func:`augment_fuel_table_pin_bu`: every
    existing row and column is kept **byte-for-byte** and only the seven pin-cell
    geometry columns are appended.  The geometry is harvested for the resolvable
    ``(library_id, type_id)`` keys (via :func:`build_fuel_table` in memory) and
    left-joined onto the on-disk store; any key with no harvestable dec inp
    (paramA-without-deck / legacy / ga80) keeps ``NaN`` geometry columns — the
    same NaN contract as ``zone_pin_count``.

    Idempotent: re-running drops and re-derives the geometry columns.  When
    ``persist`` the result is written atomically back to ``paths.store``.
    """
    paths = _as_fuel_paths(cfg_or_paths)
    existing = pd.read_parquet(paths.store)
    fresh = build_fuel_table(paths, persist=False)

    cols = list(GEOM_COLUMNS)
    key = ["library_id", "type_id"]
    geom = fresh[key + cols].drop_duplicates(subset=key)
    base = existing.drop(columns=[c for c in cols if c in existing.columns])
    merged = base.merge(geom, on=key, how="left")           # left-join preserves order
    for c in cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")
    ordered = [c for c in SCHEMA_COLUMNS if c in merged.columns]
    ordered += [c for c in merged.columns if c not in ordered]
    merged = merged[ordered]

    if len(merged) != len(existing):
        raise ValueError(
            f"geometry augment changed row count {len(existing)} -> {len(merged)}"
        )
    if persist:
        _atomic_write_parquet(paths.store, merged)
    return merged


def augment_fuel_table_kinf_shape(cfg_or_paths: Any, *, persist: bool = True
                                  ) -> pd.DataFrame:
    """Additively add the :data:`KCONV_SHAPE_COLUMNS` to an existing fuel parquet.

    Same preserve-everything contract as :func:`augment_fuel_table_pin_bu` /
    :func:`augment_fuel_table_geometry`: every existing row and column is kept
    **byte-for-byte** and only the nine k-conv curve-shape columns are appended.
    The shape features are harvested for the resolvable ``(library_id, type_id)``
    keys (via :func:`build_fuel_table` in memory — so pass a config/paths whose
    ``paramA_root`` resolves the same paramA package the store was built from, or
    those rows keep ``NaN``) and left-joined onto the on-disk store; any key with
    no harvestable reference curve (legacy / dehydrated ga80) keeps ``NaN`` — the
    same contract as ``kinf0``.

    Idempotent: re-running drops and re-derives the shape columns.  When
    ``persist`` the result is written atomically back to ``paths.store``.
    """
    paths = _as_fuel_paths(cfg_or_paths)
    existing = pd.read_parquet(paths.store)
    fresh = build_fuel_table(paths, persist=False)

    cols = list(KCONV_SHAPE_COLUMNS)
    key = ["library_id", "type_id"]
    shape = fresh[key + cols].drop_duplicates(subset=key)
    base = existing.drop(columns=[c for c in cols if c in existing.columns])
    merged = base.merge(shape, on=key, how="left")          # left-join preserves order
    for c in cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")
    ordered = [c for c in SCHEMA_COLUMNS if c in merged.columns]
    ordered += [c for c in merged.columns if c not in ordered]
    merged = merged[ordered]

    if len(merged) != len(existing):
        raise ValueError(
            f"kinf-shape augment changed row count {len(existing)} -> {len(merged)}"
        )
    if persist:
        _atomic_write_parquet(paths.store, merged)
    return merged


# --------------------------------------------------------------------------- #
# pair helpers
# --------------------------------------------------------------------------- #
def pair_e_core(type_a: FuelVec, type_b: FuelVec, split: float) -> float:
    """U-mass-weighted mean enrichment of a two-type feed split.

    ``split`` is the fraction of feed positions carrying ``type_a`` (0..1); the
    remaining ``1-split`` carry ``type_b``.  Each position contributes its own
    ``u_mass_g`` grams of uranium, so the core-average enrichment is

        e_core = (s*m_a*e_a + (1-s)*m_b*e_b) / (s*m_a + (1-s)*m_b).

    Falls back to a plain count-weighted mean when a U mass is unknown.
    """
    if not 0.0 <= split <= 1.0:
        raise ValueError(f"split must be in [0, 1], got {split}")
    ea, eb = type_a.u_avg_enrichment, type_b.u_avg_enrichment
    if ea is None or eb is None or math.isnan(ea) or math.isnan(eb):
        raise ValueError("both fuel types must have a known u_avg_enrichment")
    ma, mb = type_a.u_mass_g, type_b.u_mass_g
    if ma is None or mb is None or math.isnan(ma) or math.isnan(mb):
        return split * ea + (1.0 - split) * eb
    wa = split * ma
    wb = (1.0 - split) * mb
    return (wa * ea + wb * eb) / (wa + wb)


def mix_e_core(vecs: "Sequence[FuelVec]",
               fracs: "Sequence[float] | None" = None) -> float:
    """U-mass-weighted mean enrichment of an ``n``-type feed composition.

    The n-type generalization of :func:`pair_e_core` (which stays the 2-type
    entry point so every existing caller is byte-identical): with feed fractions
    ``f_i`` summing to 1 and per-assembly U masses ``m_i``,

        e_core = sum_i f_i m_i e_i / sum_i f_i m_i,

    falling back — all-or-nothing, exactly like :func:`pair_e_core` — to the
    plain fraction-weighted mean when ANY U mass is unknown.  ``fracs=None``
    means an equal split over the supplied types (the DoE-cell nominal).

    This is what a *graded* (3-type) case needs: the nominal e_core of ``A_B_C``
    is the composition mean of three enrichments, not of the first two.
    """
    if not vecs:
        raise ValueError("at least one fuel type is required")
    n = len(vecs)
    weights = [1.0 / n] * n if fracs is None else [float(f) for f in fracs]
    if len(weights) != n:
        raise ValueError(f"fracs has {len(weights)} entries for {n} types")
    if any(w < 0.0 for w in weights):
        raise ValueError(f"feed fractions must be non-negative, got {weights}")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("feed fractions must not be all zero")
    weights = [w / total for w in weights]

    enrs: list[float] = []
    for vec in vecs:
        e = vec.u_avg_enrichment
        if e is None or math.isnan(e):
            raise ValueError("every fuel type must have a known u_avg_enrichment")
        enrs.append(float(e))
    masses: list[float] = []
    for vec in vecs:
        m = vec.u_mass_g
        if m is None or (isinstance(m, float) and math.isnan(m)):
            return sum(w * e for w, e in zip(weights, enrs, strict=True))
        masses.append(float(m))
    wsum = sum(w * m for w, m in zip(weights, masses, strict=True))
    return sum(w * m * e for w, m, e in
               zip(weights, masses, enrs, strict=True)) / wsum


def case_e_core(fuel_library: "FuelLibrary", members: "Sequence[str]",
                library_id: str | None = None,
                fracs: "Sequence[float] | None" = None) -> float:
    """Nominal e_core of a case's fresh-type alphabet (2..5 members).

    A 2-member case is delegated to ``fuel_library.pair_e_core(a, b, split,
    library_id)`` — the *same call* every pair site made before this function
    existed — so the pair result is byte-identical AND a duck-typed stand-in that
    implements only ``pair_e_core`` keeps working.  3+ members resolve their
    :class:`FuelVec`\\ s and go through :func:`mix_e_core`, which is written over
    an arbitrary-length composition (no member-count cap of its own).

    Raises ``KeyError`` / ``ValueError`` on an unresolvable member, which is what
    the resolver / campaign / produce call sites already catch.
    """
    names = list(members)
    if len(names) < 2:
        raise ValueError(f"a case needs at least 2 fresh types, got {names}")
    if len(names) == 2:
        split = 0.5 if fracs is None else float(fracs[0])
        return fuel_library.pair_e_core(names[0], names[1], split, library_id)
    vecs = [fuel_library.get(name, library_id) for name in names]
    return mix_e_core(vecs, fracs)


# --------------------------------------------------------------------------- #
# core-average enrichment / feed spread — ONE recipe shared by extraction (store
# fill) and inference featurization (serve-time reconstruction).  Keeping a
# single implementation guarantees a served pattern reconstructs the exact
# ``e_core`` / ``e_split`` that extraction wrote into its store row (train/serve
# parity, plan sec. 4.4).
# --------------------------------------------------------------------------- #
def resolve_type_id(fuel_library: "FuelLibrary", library_id: str,
                    raw: str) -> str | None:
    """Resolve a batch name to a library type id, tolerant of zero padding.

    Tries the raw name, then the un-padded (``C01 -> C1``) and padded
    (``C1 -> C01``) forms against the library roster.  Returns ``None`` when the
    name resolves to no type in ``library_id``.
    """
    try:
        roster = set(fuel_library.types(library_id))
    except Exception:
        roster = set()
    if raw in roster:
        return raw
    stripped = re.sub(r"^([A-Za-z]+)0(\d)$", r"\1\2", raw)     # C01 -> C1
    if stripped in roster:
        return stripped
    padded = re.sub(r"^([A-Za-z]+)(\d)$", r"\g<1>0\2", raw)    # C1 -> C01
    if padded in roster:
        return padded
    return None


def core_enrichment_split(fuel_library: "FuelLibrary", library_id: str,
                          batch_feed: dict[str, int], *, resolve_type=None
                          ) -> tuple[float | None, float | None]:
    """U-mass-weighted core-average enrichment + enrichment spread of a feed.

    ``batch_feed`` maps each fresh batch id to its full-core position count
    (``Pattern.batch_feed()``).  The mean is U-mass weighted when *every* fed
    type carries a known ``u_mass_g`` and count weighted otherwise (all-or-
    nothing, matching the extraction recipe).  Returns ``(None, None)`` when any
    fed type is unresolvable or lacks an enrichment, so callers fall back
    identically at extraction and inference time.

    ``resolve_type(library_id, raw) -> type_id | None`` may be supplied so a
    caller can share a cached roster resolver (extraction hot loop); when omitted
    a self-contained :func:`resolve_type_id` closure is used.
    """
    if resolve_type is None:
        def resolve_type(lib: str, raw: str) -> str | None:
            return resolve_type_id(fuel_library, lib, raw)

    entries: list[tuple[int, float, float | None]] = []
    for batch, count in batch_feed.items():
        tid = resolve_type(library_id, batch)
        if tid is None:
            return None, None
        try:
            vec = fuel_library.get(tid, library_id)
        except KeyError:
            return None, None
        enr = vec.u_avg_enrichment
        if enr is None or (isinstance(enr, float) and math.isnan(enr)):
            return None, None
        mass = vec.u_mass_g
        if mass is not None and isinstance(mass, float) and math.isnan(mass):
            mass = None
        entries.append((count, float(enr), None if mass is None else float(mass)))
    if not entries:
        return None, None
    if any(m is None for _, _, m in entries):
        wsum = sum(c for c, _, _ in entries)
        e_core = sum(c * e for c, e, _ in entries) / wsum
    else:
        wsum = sum(c * m for c, _, m in entries)               # type: ignore[misc]
        e_core = sum(c * m * e for c, e, m in entries) / wsum   # type: ignore[misc]
    enrs = [e for _, e, _ in entries]
    return float(e_core), float(max(enrs) - min(enrs))


def fresh_fuel_charge(fuel_library: "FuelLibrary", library_id: str,
                      batch_feed: dict[str, int], *, resolve_type=None
                      ) -> tuple[float | None, bool]:
    """Total fresh U-235 charge of a feed — the min_fuel_cost economics metric.

    ``FE = Σ_fresh (u_mass × enrichment)`` over every fresh full-core position,
    the "total fresh U-235 charge" the ``min_fuel_cost`` objective minimizes: it
    trades enrichment (w/o) against feed (position count) exactly as the user
    directive requires.  ``batch_feed`` maps each fresh batch id to its full-core
    position count (``Pattern.batch_feed()``), so a position carrying type ``t``
    contributes ``count · u_mass_g(t) · e(t)``.

    Returns ``(charge, mass_weighted)``.  ``mass_weighted`` follows the SAME
    all-or-nothing rule as :func:`core_enrichment_split` so a run's metric is on
    ONE consistent scale:

    * every fed type has a known ``u_mass_g`` → true physical charge in grams of
      U-235: ``FE = Σ count · u_mass_g · e/100`` (``mass_weighted=True``);
    * any fed type lacks ``u_mass_g`` (e.g. the ga80 letter library, whose
      ``u_mass_g`` is all-NaN while enrichment anchors are known) → the
      count-weighted proxy ``FE = Σ count · e`` in [position·w/o] units
      (``mass_weighted=False``).  Under the near-constant per-assembly U loading
      of a fixed lattice family this is proportional to the true U-235 charge, so
      the objective's enrichment↓/feed↓ trade is preserved.

    Returns ``(None, False)`` when any fed type is unresolvable or lacks an
    enrichment — callers then fall back identically (mirrors
    :func:`core_enrichment_split`).  Lower ``FE`` is more fuel-economical.
    """
    if resolve_type is None:
        def resolve_type(lib: str, raw: str) -> str | None:
            return resolve_type_id(fuel_library, lib, raw)

    entries: list[tuple[int, float, float | None]] = []
    for batch, count in batch_feed.items():
        tid = resolve_type(library_id, batch)
        if tid is None:
            return None, False
        try:
            vec = fuel_library.get(tid, library_id)
        except KeyError:
            return None, False
        enr = vec.u_avg_enrichment
        if enr is None or (isinstance(enr, float) and math.isnan(enr)):
            return None, False
        mass = vec.u_mass_g
        if mass is not None and isinstance(mass, float) and math.isnan(mass):
            mass = None
        entries.append((count, float(enr), None if mass is None else float(mass)))
    if not entries:
        return None, False
    if any(m is None for _, _, m in entries):
        charge = sum(c * e for c, e, _ in entries)                       # position·w/o
        return float(charge), False
    charge = sum(c * m * e / 100.0 for c, e, m in entries)               # grams U-235
    return float(charge), True


# --------------------------------------------------------------------------- #
# library accessor
# --------------------------------------------------------------------------- #
def _vec_from_row(row: pd.Series) -> FuelVec:
    def _f(name: str) -> float | None:
        if name not in row:                        # tolerate pre-cond_v4 parquets
            return None
        v = row[name]
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    n_gd = row["n_gd"]
    n_gd = None if pd.isna(n_gd) else int(n_gd)
    flags = row["source_flags"]
    flags = list(flags) if flags is not None else []
    zone = row["axial_zone"]
    zone = None if zone is None or (isinstance(zone, float) and math.isnan(zone)) else str(zone)
    return FuelVec(
        library_id=str(row["library_id"]),
        type_id=str(row["type_id"]),
        u_avg_enrichment=_f("u_avg_enrichment"),
        enr_main=_f("enr_main"),
        enr_zone=_f("enr_zone"),
        u_mass_g=_f("u_mass_g"),
        n_gd=n_gd,
        gd_wt=_f("gd_wt"),
        gd_u_enr=_f("gd_u_enr"),
        axial_zone=zone,
        source_flags=flags,
        feature_poor=bool(row["feature_poor"]),
        kinf0=_f("kinf0"), kinf10=_f("kinf10"), kinf20=_f("kinf20"),
        kinf30=_f("kinf30"), bu_k1=_f("bu_k1"),
        ff_pin_max=_f("ff_pin_max"),
        xs_d1=_f("xs_d1"), xs_d2=_f("xs_d2"),
        xs_a1=_f("xs_a1"), xs_a2=_f("xs_a2"),
        xs_nf1=_f("xs_nf1"), xs_nf2=_f("xs_nf2"), xs_s12=_f("xs_s12"),
        adf_face_g1=_f("adf_face_g1"), adf_face_g2=_f("adf_face_g2"),
        adf_corner_g1=_f("adf_corner_g1"), adf_corner_g2=_f("adf_corner_g2"),
        boron_worth=_f("boron_worth"), doppler_coef=_f("doppler_coef"),
        mtc_dmod=_f("mtc_dmod"), cr1_worth=_f("cr1_worth"),
        zone_pin_count=_f("zone_pin_count"),
        pin_bu_r_inf=_f("pin_bu_r_inf"), pin_bu_paramA=_f("pin_bu_paramA"),
        pin_bu_ratio_asym=_f("pin_bu_ratio_asym"), pin_bu_bu_max=_f("pin_bu_bu_max"),
        pin_bu_n_pts=_f("pin_bu_n_pts"),
        pin_pitch=_f("pin_pitch"), asm_pitch=_f("asm_pitch"),
        r_pellet=_f("r_pellet"), r_clad_in=_f("r_clad_in"),
        r_clad_out=_f("r_clad_out"), p_over_d=_f("p_over_d"),
        v_mod_over_v_fuel=_f("v_mod_over_v_fuel"),
        kinf_dip=_f("kinf_dip"), bu_dip_gwd=_f("bu_dip_gwd"),
        kinf_peak=_f("kinf_peak"), bu_peak_gwd=_f("bu_peak_gwd"),
        reactivity_swing_pcm=_f("reactivity_swing_pcm"),
        rho_boc_minus_peak_pcm=_f("rho_boc_minus_peak_pcm"),
        depletion_slope_pcm_per_gwd=_f("depletion_slope_pcm_per_gwd"),
        kinf_eol50=_f("kinf_eol50"), kconv_is_monotone=_f("kconv_is_monotone"),
    )


class FuelLibrary:
    """Keyed accessor over the fuel-feature table.

    The key is ``(library_id, type_id)``; the same type id in different
    libraries is a *different* physical design.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)
        self._index: dict[tuple[str, str], FuelVec] = {}
        for _, row in self._df.iterrows():
            vec = _vec_from_row(row)
            self._index[vec.key] = vec

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_parquet(cls, path: str | Path) -> "FuelLibrary":
        return cls(pd.read_parquet(path))

    @classmethod
    def build(cls, cfg_or_paths: Any, *, persist: bool = True) -> "FuelLibrary":
        return cls(build_fuel_table(cfg_or_paths, persist=persist))

    # -- accessors ---------------------------------------------------------- #
    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    def libraries(self) -> list[str]:
        return sorted(self._df["library_id"].unique().tolist())

    def types(self, library_id: str) -> list[str]:
        sub = self._df[self._df["library_id"] == library_id]
        return sub["type_id"].tolist()

    def get(self, type_id: str, library_id: str) -> FuelVec:
        """Return the FuelVec for ``(library_id, type_id)``.

        Raises a hard :class:`KeyError` with a helpful message on an unknown
        key (listing the libraries that *do* carry ``type_id``, if any).
        """
        key = (library_id, type_id)
        vec = self._index.get(key)
        if vec is not None:
            return vec
        libs_known = self.libraries()
        also_in = sorted(
            lib for lib in libs_known if (lib, type_id) in self._index
        )
        hint = (
            f" (type {type_id!r} exists in library/libraries {also_in})"
            if also_in else
            f" (library {library_id!r} carries types {self.types(library_id)[:12]}...)"
            if library_id in libs_known else
            f" (unknown library {library_id!r}; known libraries: {libs_known})"
        )
        raise KeyError(
            f"no fuel row for (library_id={library_id!r}, type_id={type_id!r}){hint}"
        )

    def library_enrichment_range(self, library_id: str) -> tuple[float, float] | None:
        """Min/max U-average enrichment over a library (None if all unknown)."""
        sub = self._df[self._df["library_id"] == library_id]
        vals = pd.to_numeric(sub["u_avg_enrichment"], errors="coerce").dropna()
        if vals.empty:
            return None
        return float(vals.min()), float(vals.max())

    def pair_e_core(self, type_a: str | FuelVec, type_b: str | FuelVec,
                    split: float, library_id: str | None = None) -> float:
        """U-mass-weighted mean enrichment of a two-type feed split.

        ``type_a`` / ``type_b`` may be :class:`FuelVec` objects or type-id
        strings (resolved within ``library_id``).
        """
        va = type_a if isinstance(type_a, FuelVec) else self.get(type_a, library_id)
        vb = type_b if isinstance(type_b, FuelVec) else self.get(type_b, library_id)
        return pair_e_core(va, vb, split)

    def case_e_core(self, members: Sequence[str], library_id: str | None = None,
                    fracs: Sequence[float] | None = None) -> float:
        """Nominal e_core of a 2..5-type case alphabet (:func:`case_e_core`)."""
        return case_e_core(self, members, library_id, fracs)
