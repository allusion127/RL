"""Pre-campaign GEOMETRY VALIDATION PROTOCOL (readiness review sec. 4c).

Gates the whole pin-pitch / pin-radius optimization axis on a DeCART->MASTER
*transfer test* BEFORE any optimizer consumes geometry-varied fuel types.  For a
grid of admissible pin-pitch / pin-radius variants over a few enrichment/Gd anchors
it:

1. **generates** variant dec decks (material anchor edit + GEOM pin-pitch / cellgeo
   radii edit via :func:`lpopt.design.lattice.edit_dec_geom_text`, which asserts the
   assembly-pitch envelope and guide tubes stay frozen);
2. **runs DeCART** on them (cap 4 parallel — the existing lattice chain), skipped in
   ``--dry-run``;
3. **harvests** each variant's physics + the new geometry columns into a SIDE table
   (never the main ``fuel_types`` until a human accepts it);
4. **blind-probes** ``probe_size`` (16) MASTER chains per variant through the
   standard verifier into the registered scratch store
   ``C:/Users/USER/AppData/Local/Temp/eqlp_geomchk``, scoring the CURRENT champion's
   blind prediction (no fine-tune) against MASTER truth;
5. **scores acceptance bands** (the median healthy blind-transfer levels of
   ``data/curriculum/transfer_curve.json``) per target and writes a verdict report.

A variant that clears every band is transfer-safe and may enter the optimizer with
the serve-time OOD guard (sec. 4b) active; a variant that fails -> **quarantine the
geometry axis** until a geometry-aware v5 retrain.

``--dry-run`` runs the whole pipeline with NO DeCART and NO MASTER: variant decks
are still generated + parsed (the geometry harvest is real), the side table's
physics columns are seeded from the anchor (radius variants get a ``u_mass ~ r^2``
shift so the OOD guard is exercised), and the blind probe scores a deterministic
stub champion against the deterministic :class:`~lpopt.search.stub.StubEvaluator`.
It is the DeCART-less end-to-end smoke.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..data.fuel_types import (
    FuelLibrary,
    FuelVec,
    GEOM_COLUMNS,
    NOMINAL_ASM_PITCH,
    geom_derived,
    parse_dec_geom,
)
from ..search.genome import fresh_units_from_feed, random_genome
from ..vendor.masterrl.domain import CaseKey, Pattern
from .lattice import LatticeError, edit_dec_geom_text, edit_dec_text, resolve_template
from .spec import ANCHOR_DESIGNS, DesignRegistry, FuelDesign

#: Registered scratch workspace for geometry validation (an additional working dir).
DEFAULT_SCRATCH = Path(r"C:/Users/USER/AppData/Local/Temp/eqlp_geomchk")

#: Nominal pin-cell geometry (constant across every training lattice).
NOMINAL_PIN_PITCH = 1.285
NOMINAL_R_PELLET = 0.4096
NOMINAL_R_CLAD_IN = 0.4178
NOMINAL_R_CLAD_OUT = 0.4750
NOMINAL_NPINS = 16

#: Admissible pin-pitch ceiling (review sec. 1.1): the pin array must fit inside the
#: fixed assembly envelope, ``npins * pin_pitch <= asm_pitch`` -> pitch <= 1.29858 cm
#: (+1.057 %).  A symmetric +-3 % pitch grid is NOT admissible upward.
PITCH_CEIL_FRAC = (NOMINAL_ASM_PITCH / NOMINAL_NPINS) / NOMINAL_PIN_PITCH - 1.0

#: Acceptance bands (review sec. 4c step 6): ``target -> (min_spearman, max_mae)``.
#: Tied to the median healthy blind-transfer levels; the ring-0 cold-start outlier
#: is excluded as the floor reference.
ACCEPTANCE_BANDS: dict[str, tuple[float, float]] = {
    "f_r": (0.70, 0.5),
    "cyclen": (0.60, 15.0),
    "cbc_max": (0.60, 50.0),
}
#: ``target -> (surrogate column, MASTER FOM attribute)``.
_PROBE_TARGETS: tuple[tuple[str, int, str], ...] = (
    ("f_r", 0, "f_r"),
    ("cbc_max", 1, "cbc_max"),
    ("cyclen", 3, "cyclen"),
)


class GeomCheckError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# variant grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeomVariant:
    """One pin-geometry variant of an enrichment/Gd anchor."""

    alias: str
    design: FuelDesign
    pitch_frac: float
    radius_frac: float

    @property
    def pin_pitch(self) -> float:
        return NOMINAL_PIN_PITCH * (1.0 + self.pitch_frac)

    @property
    def r_pellet(self) -> float:
        return NOMINAL_R_PELLET * (1.0 + self.radius_frac)

    @property
    def r_clad_in(self) -> float:
        return NOMINAL_R_CLAD_IN * (1.0 + self.radius_frac)

    @property
    def r_clad_out(self) -> float:
        return NOMINAL_R_CLAD_OUT * (1.0 + self.radius_frac)

    @property
    def type_id(self) -> str:
        """Descriptive variant id, e.g. ``P5851Z1G06N12_P+005_R-030``."""
        return (f"{self.design.type_id}"
                f"_P{_pct(self.pitch_frac)}_R{_pct(self.radius_frac)}")

    def geometry_columns(self) -> dict[str, float]:
        """The seven :data:`GEOM_COLUMNS` for this variant."""
        d = geom_derived(self.pin_pitch, self.r_pellet, self.r_clad_out)
        return {
            "pin_pitch": self.pin_pitch,
            "asm_pitch": NOMINAL_ASM_PITCH,
            "r_pellet": self.r_pellet,
            "r_clad_in": self.r_clad_in,
            "r_clad_out": self.r_clad_out,
            "p_over_d": d.get("p_over_d", float("nan")),
            "v_mod_over_v_fuel": d.get("v_mod_over_v_fuel", float("nan")),
        }

    def admissibility(self) -> str | None:
        """A rejection reason if this variant is geometrically inadmissible, else None."""
        if self.pitch_frac > PITCH_CEIL_FRAC + 1e-9:
            return (f"pin_pitch +{self.pitch_frac*100:.2f}% exceeds the +{PITCH_CEIL_FRAC*100:.2f}% "
                    f"assembly-envelope ceiling (rods would fill the node)")
        if self.r_clad_out >= self.pin_pitch / 2.0:
            return (f"clad OD {2*self.r_clad_out:.4f} >= pitch {self.pin_pitch:.4f} "
                    f"(rods touch within the lattice)")
        return None


def _pct(frac: float) -> str:
    sign = "+" if frac >= 0 else "-"
    return f"{sign}{abs(int(round(frac * 1000))):03d}"        # +1.0% -> "+010"


def geom_variant_grid(
    pitch_fracs: Sequence[float],
    radius_fracs: Sequence[float],
    anchors: Sequence[FuelDesign],
    registry: DesignRegistry | None = None,
) -> list[GeomVariant]:
    """The ``pitch x radius x anchor`` variant grid (inadmissible cells dropped).

    ``registry`` (optional) assigns each anchor a stable 2-char MASTER alias so the
    variant deck / product names stay stable across re-runs.
    """
    registry = registry or DesignRegistry()
    out: list[GeomVariant] = []
    for anchor in anchors:
        base_alias = registry.alias(anchor)
        for pf in pitch_fracs:
            for rf in radius_fracs:
                v = GeomVariant(alias=base_alias, design=anchor,
                                pitch_frac=float(pf), radius_frac=float(rf))
                if v.admissibility() is None:
                    out.append(v)
    return out


# --------------------------------------------------------------------------- #
# deck generation
# --------------------------------------------------------------------------- #
def _variant_caseid(variant: GeomVariant, idx: int) -> str:
    """A <=5-char DeCART/MASTER CASEID for a variant (``G`` + 4 digits)."""
    return f"G{idx:04d}"


def write_variant_deck(variant: GeomVariant, template_text: str, out_dir: Path,
                       caseid: str) -> Path:
    """Write the variant dec deck: anchor material edit + GEOM pin-geometry edit.

    Raises :class:`LatticeError` (from the GEOM editor's frozen-envelope guard) if
    the edit would move the assembly pitch or a guide-tube cellgeo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    text = edit_dec_text(template_text, variant.design, caseid)
    text = edit_dec_geom_text(
        text, pin_pitch=variant.pin_pitch, r_pellet=variant.r_pellet,
        r_clad_in=variant.r_clad_in, r_clad_out=variant.r_clad_out,
    )
    deck = out_dir / f"dec_{caseid}.inp"
    deck.write_text(text, encoding="utf-8")
    return deck


def generate_variant_decks(variants: Sequence[GeomVariant], apr1400_root: str | Path,
                           deck_root: Path, log: Callable[[str], None] = print
                           ) -> list[dict]:
    """Generate one dec deck per variant; returns per-variant deck records.

    Each record carries the written ``deck`` path, the ``caseid``, and the harvested
    ``geom`` columns re-parsed from the deck (a round-trip check that the GEOM edit
    landed).  A variant whose template is missing or whose edit trips the frozen
    guard is recorded with an ``error`` (never aborts the batch).
    """
    apr1400_root = Path(apr1400_root)
    recs: list[dict] = []
    for idx, v in enumerate(variants):
        caseid = _variant_caseid(v, idx)
        rec: dict[str, Any] = {"variant": v, "caseid": caseid, "type_id": v.type_id}
        try:
            template = resolve_template(v.design, apr1400_root)
            deck = write_variant_deck(v, template.read_text(encoding="utf-8", errors="replace"),
                                      deck_root / caseid, caseid)
            rec["deck"] = deck
            rec["geom"] = parse_dec_geom(deck)
        except (LatticeError, OSError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            log(f"[geomcheck] deck gen FAILED for {v.type_id}: {rec['error']}")
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- #
# side table (never the main fuel_types until accepted)
# --------------------------------------------------------------------------- #
def _anchor_vec(base_fuel: FuelLibrary, design: FuelDesign) -> FuelVec | None:
    """A representative in-population FuelVec for a design's (gd_wt, n_gd, zoning)
    so the side-table variant inherits plausible physics in dry-run.

    Prefers a lattice type whose harvested geometry is present (260624/5.8_5.1/CPHA);
    falls back to the first such type, else None.
    """
    df = base_fuel.frame
    have_geom = df[df["pin_pitch"].notna()] if "pin_pitch" in df.columns else df.iloc[0:0]
    for _, row in have_geom.iterrows():
        try:
            return base_fuel.get(str(row["type_id"]), str(row["library_id"]))
        except KeyError:
            continue
    return None


def side_table_row(variant: GeomVariant, deck_rec: dict, base_fuel: FuelLibrary,
                   *, dry_run: bool) -> dict:
    """One side-table row for a variant.

    Geometry columns are the (real) harvested/edited geometry.  Physics columns come
    from DeCART products when present (live), else are seeded from the anchor with a
    ``u_mass ~ (r_pellet/r_base)^2`` radius shift so the dry-run exercises the OOD
    guard's radius canary.
    """
    geom = deck_rec.get("geom") or variant.geometry_columns()
    # ``probe_key`` (the short caseid, e.g. "G0000") is what the blind-probe pattern
    # carries as its fresh-batch name and what the augmented fuel library keys the
    # variant under, so a REAL champion resolves each variant's OWN physics (the
    # descriptive ``type_id`` has underscores and would break the pair split).
    row: dict[str, Any] = {"library_id": "geomchk", "type_id": variant.type_id,
                           "probe_key": str(deck_rec.get("caseid") or variant.alias),
                           "pitch_frac": variant.pitch_frac,
                           "radius_frac": variant.radius_frac}
    for c in GEOM_COLUMNS:
        row[c] = float(geom.get(c, float("nan")))

    # LIVE: real physics harvested from the variant's DeCART products (review sec.
    # 4c step 3 — harvest_lattice_cond_v4 + the geometry columns) overrides the seed.
    harvested = deck_rec.get("harvest") or {}

    anchor = _anchor_vec(base_fuel, variant.design)
    if anchor is not None:
        for attr in ("u_avg_enrichment", "enr_main", "enr_zone", "gd_u_enr",
                     "u_mass_g", "kinf0", "kinf10", "kinf20", "kinf30", "bu_k1",
                     "ff_pin_max", "boron_worth", "doppler_coef", "mtc_dmod",
                     "cr1_worth", "zone_pin_count", "xs_a2", "xs_nf2", "xs_s12",
                     "adf_corner_g2"):
            val = getattr(anchor, attr, None)
            row[attr] = float("nan") if val is None else float(val)
        row["n_gd"] = anchor.n_gd
        # radius shift on the U inventory (per-pellet mass ~ r_pellet^2).
        base_rp = getattr(anchor, "r_pellet", None) or NOMINAL_R_PELLET
        if row.get("u_mass_g") == row.get("u_mass_g"):      # not NaN
            row["u_mass_g"] = float(row["u_mass_g"]) * (variant.r_pellet / base_rp) ** 2
    # real harvested physics wins when present (live run).
    for k, val in harvested.items():
        if k in GEOM_COLUMNS:
            continue                                        # geometry comes from the deck
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            row[k] = float(val)
    row["dry_run"] = bool(dry_run) and not harvested
    return row


def side_table_vec(row: dict) -> FuelVec:
    """A :class:`FuelVec` for a side-table row (fed to the champion + OOD guard)."""
    def _f(name: str) -> float | None:
        v = row.get(name)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(fv) else fv

    return FuelVec(
        library_id="geomchk", type_id=str(row["type_id"]),
        u_avg_enrichment=_f("u_avg_enrichment"), enr_main=_f("enr_main"),
        enr_zone=_f("enr_zone"), gd_u_enr=_f("gd_u_enr"), u_mass_g=_f("u_mass_g"),
        n_gd=(None if row.get("n_gd") is None else int(row["n_gd"])),
        kinf0=_f("kinf0"), kinf10=_f("kinf10"), kinf20=_f("kinf20"),
        kinf30=_f("kinf30"), bu_k1=_f("bu_k1"), ff_pin_max=_f("ff_pin_max"),
        boron_worth=_f("boron_worth"), doppler_coef=_f("doppler_coef"),
        mtc_dmod=_f("mtc_dmod"), cr1_worth=_f("cr1_worth"),
        zone_pin_count=_f("zone_pin_count"), xs_a2=_f("xs_a2"),
        xs_nf2=_f("xs_nf2"), xs_s12=_f("xs_s12"), adf_corner_g2=_f("adf_corner_g2"),
        pin_pitch=_f("pin_pitch"), asm_pitch=_f("asm_pitch"),
        r_pellet=_f("r_pellet"), r_clad_in=_f("r_clad_in"),
        r_clad_out=_f("r_clad_out"), p_over_d=_f("p_over_d"),
        v_mod_over_v_fuel=_f("v_mod_over_v_fuel"),
    )


# --------------------------------------------------------------------------- #
# blind probe
# --------------------------------------------------------------------------- #
class StubProbeModel:
    """Deterministic stand-in champion for the DeCART-less dry-run.

    ``predict`` returns a 7-column mean whose f_r/cbc_max/cyclen track the
    :class:`~lpopt.search.stub.StubEvaluator` FOM (plus a tiny deterministic per-
    pattern perturbation), so the blind-probe scoring pipeline exercises a realistic
    high-Spearman / low-MAE PASS without a trained ensemble or torch.
    """

    def __init__(self, noise: float = 0.02) -> None:
        from ..search.stub import StubEvaluator
        self._stub = StubEvaluator()
        self.noise = float(noise)

    def predict(self, patterns: Sequence[Pattern], cases: Sequence[CaseKey]):
        import numpy as np
        cases = list(cases)
        mean = np.full((len(patterns), 7), np.nan)
        for i, pat in enumerate(patterns):
            feed = int(cases[i].feed) if i < len(cases) else 121
            fom = self._stub.fom_for(pat.digest, feed)
            j = _jitter(pat.digest)
            mean[i, 0] = fom.f_r * (1.0 + self.noise * j)
            mean[i, 1] = fom.cbc_max * (1.0 + self.noise * j)
            mean[i, 3] = fom.cyclen * (1.0 + self.noise * j)
        return _MeanOnly(mean)


@dataclass
class _MeanOnly:
    mean: Any


def _jitter(digest: str) -> float:
    """Deterministic perturbation in [-1, 1) from a digest."""
    return (int(digest[:8], 16) / float(1 << 32)) * 2.0 - 1.0


def build_probe_patterns(pair: str, feed: int, n: int, rng: random.Random
                         ) -> list[Pattern]:
    """``n`` distinct random loading patterns for a ``pair`` at ``feed`` (blind set)."""
    n_fresh = fresh_units_from_feed(feed)
    allow_single = n_fresh > 30
    seen: set[str] = set()
    out: list[Pattern] = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        g = random_genome(rng, pair, n_fresh, max_shuffle_depth=2,
                          allow_single_cycle_discharge=allow_single)
        pat = g.to_pattern()
        key = pat.canonical()
        if key in seen:
            continue
        seen.add(key)
        out.append(pat)
    return out


def _stub_case_data(case: CaseKey):
    """A minimal :class:`CaseData` for a bare evaluator (StubEvaluator ignores assets)."""
    from ..vendor.masterrl.dataset import CaseData
    return CaseData(key=case, cell=0.0, records=(), template_path=Path("."),
                    restart_path=Path("."))


def run_probe_chains(evaluator: Any, patterns: Sequence[Pattern],
                     cases: Sequence[CaseKey]) -> list[Any]:
    """Run one chain per pattern through ``evaluator``; returns per-pattern FOM|None.

    ``evaluator`` may be a bare :class:`PatternEvaluator` (``evaluate(case_data,
    pattern)`` — the StubEvaluator dry-run path, asset-free) or a
    :class:`~lpopt.search.verify.WaveVerifier` (``evaluate_wave`` — the live path).
    """
    if hasattr(evaluator, "evaluate_wave"):
        from ..search.verify import WaveEntry
        entries = [WaveEntry(pattern=p, case_key=c,
                             resolved_assets=evaluator.resolver.resolve(c))
                   for p, c in zip(patterns, cases)]
        outcomes = evaluator.evaluate_wave(entries)
        return [oc.fom if oc.status == "converged" else None for oc in outcomes]
    foms: list[Any] = []
    for p, c in zip(patterns, cases):
        try:
            res = evaluator.evaluate(_stub_case_data(c), p)
            fom = getattr(res, "fom", None)
            foms.append(fom if (fom is not None and getattr(fom, "converged", True)) else None)
        except Exception:  # noqa: BLE001 — a chain error is a missing label
            foms.append(None)
    return foms


def _spearman(truth: Sequence[float], pred: Sequence[float]) -> float | None:
    if len(truth) < 3 or len(set(truth)) < 2 or len(set(pred)) < 2:
        return None
    from ..vendor.masterrl.surrogate import _spearman as vendor_spearman
    import numpy as np
    val = float(vendor_spearman(np.asarray(truth, float), np.asarray(pred, float)))
    return val if math.isfinite(val) else None


def score_variant(pred_mean, foms: Sequence[Any]) -> dict:
    """Per-target blind-transfer stats + band verdict for one variant.

    ``pred_mean`` is the champion's ``[N, 7]`` mean; ``foms`` the ``N`` MASTER truths
    (None for a non-converged chain).  Returns per-target ``spearman``/``mae``/``n``
    + ``pass`` and an overall ``verdict``.
    """
    import numpy as np
    pred_mean = np.asarray(pred_mean, float)
    idx = [i for i, f in enumerate(foms) if f is not None]
    per_target: dict[str, dict] = {}
    all_pass = bool(idx)
    for name, col, attr in _PROBE_TARGETS:
        rows = [(float(pred_mean[i, col]), float(getattr(foms[i], attr)))
                for i in idx
                if math.isfinite(pred_mean[i, col])
                and getattr(foms[i], attr, None) is not None]
        min_sp, max_mae = ACCEPTANCE_BANDS[name]
        if len(rows) < 3:
            per_target[name] = {"n": len(rows), "spearman": None, "mae": None,
                                "pass": False, "note": "insufficient converged chains"}
            all_pass = False
            continue
        preds = [r[0] for r in rows]
        truths = [r[1] for r in rows]
        sp = _spearman(truths, preds)
        mae = float(np.mean(np.abs(np.asarray(preds) - np.asarray(truths))))
        ok = sp is not None and sp >= min_sp and mae <= max_mae
        per_target[name] = {"n": len(rows), "spearman": sp, "mae": mae,
                            "pass": bool(ok), "min_spearman": min_sp, "max_mae": max_mae}
        all_pass = all_pass and ok
    return {"n_converged": len(idx), "per_target": per_target,
            "verdict": "PASS" if all_pass else "FAIL"}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
@dataclass
class GeomValidationResult:
    dry_run: bool
    scratch_dir: Path
    n_variants: int
    n_admissible: int
    variants: list[dict] = field(default_factory=list)
    overall_verdict: str = "PASS"
    side_table_path: Path | None = None
    verdict_path: Path | None = None
    ood_warnings: list[str] = field(default_factory=list)
    constraint_note: str = ""


def run_geom_validation(
    cfg: Any,
    *,
    pitch_fracs: Sequence[float],
    radius_fracs: Sequence[float],
    anchors: Sequence[FuelDesign] | None = None,
    feed: int = 121,
    probe_size: int = 16,
    dry_run: bool = True,
    scratch_dir: str | Path = DEFAULT_SCRATCH,
    model: Any = None,
    evaluator: Any = None,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> GeomValidationResult:
    """Run the geometry-validation protocol (review sec. 4c).

    In ``dry_run`` no DeCART/MASTER runs: decks are generated + parsed, the side
    table is seeded from the anchors, and the blind probe scores a stub champion
    against the StubEvaluator.  ``model`` / ``evaluator`` may be injected (a real
    ``PosValCnnBackend`` and ``WaveVerifier`` for a live run, or test doubles).
    """
    import numpy as np
    import pandas as pd

    base = cfg.source_path.parent if getattr(cfg, "source_path", None) else Path.cwd()
    d = getattr(cfg, "design", None)
    apr = Path(getattr(d, "apr1400_root", "../0_APR1400"))
    apr = apr if apr.is_absolute() else (base / apr)

    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    deck_root = scratch / "decks"

    anchors = list(anchors) if anchors else list(ANCHOR_DESIGNS)
    registry = DesignRegistry.load(scratch / "registry.json")
    variants = geom_variant_grid(pitch_fracs, radius_fracs, anchors, registry)
    registry.save(scratch / "registry.json")

    n_requested = len(anchors) * len(pitch_fracs) * len(radius_fracs)
    result = GeomValidationResult(
        dry_run=dry_run, scratch_dir=scratch,
        n_variants=n_requested, n_admissible=len(variants),
        constraint_note=_CONSTRAINT_NOTE,
    )
    if not variants:
        log("[geomcheck] no admissible variants in the requested grid")
        _write_verdict(result, scratch, log)
        return result

    log(f"[geomcheck] {len(variants)}/{n_requested} admissible variants "
        f"(pitch ceiling +{PITCH_CEIL_FRAC*100:.2f}%); dry_run={dry_run}")

    # 1-2. decks (+ DeCART in live mode)
    deck_recs = generate_variant_decks(variants, apr, deck_root, log=log)
    if not dry_run:
        _run_decart_variants(deck_recs, cfg, deck_root, log=log)

    # 3. side table
    base_fuel = _load_base_fuel(cfg, base)
    side_rows = [side_table_row(rec["variant"], rec, base_fuel, dry_run=dry_run)
                 for rec in deck_recs if "error" not in rec]
    side_df = pd.DataFrame(side_rows)
    side_path = scratch / "side_table.parquet"
    if len(side_df):
        side_df.to_parquet(side_path, index=False)
        result.side_table_path = side_path

    # OOD guard over the side table (parts 4b <-> 4c tie-in).
    ood_by_type = _side_table_ood(base_fuel, side_rows)

    # 4-5. blind probe + scoring
    the_model = model if model is not None else (StubProbeModel() if dry_run else None)
    the_eval = evaluator if evaluator is not None else (_default_evaluator(dry_run))
    if the_model is None or the_eval is None:
        raise GeomCheckError(
            "a live geom-validation needs an injected champion model + verifier; "
            "use --dry-run for the stub path")
    # A REAL champion (has a ``.fuel`` FuelLibrary) must be able to featurize the
    # variant types: swap in a library = base population + the side-table rows so the
    # blind probe scores the variant's OWN harvested physics (base rows unchanged).
    # Freeze the OOD envelope over the BASE population FIRST so a variant can never
    # stretch its own envelope, then clear the roster cache so the new library routes.
    if hasattr(the_model, "fuel") and side_rows:
        try:
            if hasattr(the_model, "feature_ood_envelope"):
                the_model.feature_ood_envelope()          # freeze over base population
            the_model.fuel = _augmented_fuel(base_fuel, side_rows)
            the_model._roster_cache = None
        except Exception as exc:  # noqa: BLE001 — never abort on the augment
            log(f"[geomcheck] fuel augment skipped ({exc}); model uses base library")

    all_pass = True
    ok_recs = [r for r in deck_recs if "error" not in r]
    for rec in ok_recs:
        v: GeomVariant = rec["variant"]
        rng = random.Random(seed + int(hashlib.sha1(v.type_id.encode()).hexdigest()[:8], 16) % 9973)
        # probe pair = the variant's short caseid (== augmented-library key), so a
        # real champion featurizes the variant's own physics; same-type core.
        pair = f"{rec['caseid']}_{rec['caseid']}"
        patterns = build_probe_patterns(pair, feed, probe_size, rng)
        cases = [CaseKey(pair, int(feed))] * len(patterns)
        pred = the_model.predict(patterns, cases)
        foms = run_probe_chains(the_eval, patterns, cases)
        # persist the blind-probe chains into the scratch store.
        chains_dir = scratch / "chains" / rec["caseid"]
        chains_dir.mkdir(parents=True, exist_ok=True)
        _dump_chains(chains_dir / "chains.json", v, patterns, pred.mean, foms)
        score = score_variant(pred.mean, foms)
        entry = {
            "type_id": v.type_id, "alias": v.alias, "caseid": rec["caseid"],
            "pitch_frac": v.pitch_frac, "radius_frac": v.radius_frac,
            "geometry": rec.get("geom", v.geometry_columns()),
            "n_probe": len(patterns), **score,
            "ood_channels": [list(cz) for cz in ood_by_type.get(v.type_id, [])],
        }
        result.variants.append(entry)
        all_pass = all_pass and (score["verdict"] == "PASS")
        log(f"[geomcheck] {v.type_id}: {score['verdict']} "
            f"({score['n_converged']}/{len(patterns)} conv) "
            + ", ".join(f"{k} Sp={_fmt(s.get('spearman'))} MAE={_fmt(s.get('mae'))}"
                        for k, s in score["per_target"].items()))

    for rec in deck_recs:
        if "error" in rec:
            v = rec["variant"]
            result.variants.append({"type_id": v.type_id, "error": rec["error"],
                                    "verdict": "ERROR"})
            all_pass = False

    result.overall_verdict = "PASS" if all_pass else "FAIL"
    result.ood_warnings = [
        f"{tid} [" + ", ".join(f"{c} z={z:+.1f}" for c, z in ood) + "]"
        for tid, ood in sorted(ood_by_type.items())
    ]
    _write_verdict(result, scratch, log)
    return result


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_CONSTRAINT_NOTE = (
    "Load-bearing constraint (review sec. 0, VERIFIED): the assembly envelope is "
    "fixed iff the SECOND pitch token stays 20.7772 (== coredeck CoreParams.wide, "
    "emitted into MASTER %GEN_GEO). Only the FIRST pitch token (pin pitch) and "
    "cellgeo 1/2 fuel-pin radii may vary; asm_pitch, npins=16, and guide-tube "
    "cellgeo 3-6 stay frozen. Under that rule the MASTER nodal core model, its mesh, "
    "and restart files remain valid (sec. 3). The GEOM deck editor asserts this on "
    "every generated variant."
)


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _load_base_fuel(cfg: Any, base: Path) -> FuelLibrary:
    store = getattr(getattr(cfg, "model", None), "store_dir", "data/store")
    sp = Path(store)
    sp = sp if sp.is_absolute() else (base / sp)
    fpath = sp / "fuel_types.parquet"
    if fpath.exists():
        return FuelLibrary.from_parquet(fpath)
    return FuelLibrary.build(cfg, persist=False)


def _default_evaluator(dry_run: bool) -> Any:
    if dry_run:
        from ..search.stub import StubEvaluator
        return StubEvaluator()
    return None


def _augmented_fuel(base_fuel: FuelLibrary, side_rows: Sequence[dict]) -> FuelLibrary:
    """A FuelLibrary = base population + the ``geomchk`` side-table variant rows.

    Used to let a REAL champion featurize a variant type from its own vec (the live
    blind probe); the base rows stay byte-identical.
    """
    import pandas as pd
    base_df = base_fuel.frame.copy()
    extra = pd.DataFrame([_row_to_frame_dict(r) for r in side_rows])
    merged = pd.concat([base_df, extra], ignore_index=True)
    return FuelLibrary(merged)


def _row_to_frame_dict(row: dict) -> dict:
    # key the augmented-library record by the SHORT probe_key so a probe pattern's
    # fresh-batch name (the caseid) resolves to this variant's vec.
    rec = side_table_vec(row).to_record()
    rec["type_id"] = str(row.get("probe_key") or rec["type_id"])
    return rec


def _side_table_ood(base_fuel: FuelLibrary, side_rows: Sequence[dict]
                    ) -> dict[str, list[tuple[str, float]]]:
    """Run the serve-time OOD guard over the side-table variants (parts 4b<->4c)."""
    from ..model.ood_guard import feature_ood_vecs, population_envelope_from_library
    # envelope over the geometry-carrying training population.
    df = base_fuel.frame
    libs = None
    if "pin_pitch" in df.columns:
        with_geom = sorted(df[df["pin_pitch"].notna()]["library_id"].unique().tolist())
        libs = with_geom or None
    env = population_envelope_from_library(base_fuel, libs)
    vecs = {str(r["type_id"]): side_table_vec(r) for r in side_rows}
    return feature_ood_vecs(vecs, env)


def _dump_chains(path: Path, variant: GeomVariant, patterns, pred_mean, foms) -> None:
    import numpy as np
    pred_mean = np.asarray(pred_mean, float)
    rows = []
    for i, pat in enumerate(patterns):
        f = foms[i]
        rows.append({
            "digest": pat.digest,
            "pred": {"f_r": _num(pred_mean[i, 0]), "cbc_max": _num(pred_mean[i, 1]),
                     "cyclen": _num(pred_mean[i, 3])},
            "truth": (None if f is None else
                      {"f_r": _num(getattr(f, "f_r", None)),
                       "cbc_max": _num(getattr(f, "cbc_max", None)),
                       "cyclen": _num(getattr(f, "cyclen", None)),
                       "converged": bool(getattr(f, "converged", True))}),
        })
    path.write_text(json.dumps({"type_id": variant.type_id, "chains": rows}, indent=2),
                    encoding="utf-8")


def _num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _run_decart_variants(deck_recs: list[dict], cfg: Any, deck_root: Path,
                         log: Callable[[str], None]) -> None:
    """Run DeCART on the generated variant decks (cap 4 parallel) and harvest.

    LIVE path only.  Mirrors :func:`lpopt.design.lattice.run_batch`'s bounded
    concurrency but on pre-written geometry decks; on completion the products'
    physics feed the side table.  DeCART-less dry-run never reaches here.
    """
    import time as _time

    from ..data.fuel_types import harvest_lattice_cond_v4
    from .lattice import DEFAULT_DECART_EXE, harvest, launch_decart

    d = getattr(cfg, "design", None)
    exe = getattr(d, "decart_exe", DEFAULT_DECART_EXE)
    max_parallel = int(getattr(d, "max_parallel", 4))
    timeout_s = float(getattr(d, "decart_timeout", 5400))
    poll_s = 15.0

    pending = [r for r in deck_recs if "deck" in r]
    active: list[tuple[dict, Any]] = []
    while pending or active:
        while pending and len(active) < max_parallel:
            rec = pending.pop(0)
            run = launch_decart(rec["deck"], deck_root / rec["caseid"],
                                rec["variant"].design, rec["caseid"], exe=exe)
            active.append((rec, run))
        _time.sleep(poll_s)
        still = []
        for rec, run in active:
            timed_out = (time.monotonic() - run.started) > timeout_s
            if run.poll():
                harvest(run)
                rec["decart_hgc"] = run.hgc_path
                # harvest the variant's real physics (kinf curve + branch/xs/adf)
                # from the products beside the deck (review sec. 4c step 3).
                anchor_out = (deck_root / rec["caseid"] / f"FA_{rec['caseid']}.out")
                try:
                    rec["harvest"], _flags = harvest_lattice_cond_v4(anchor_out)
                except (OSError, ValueError):
                    rec["harvest"] = {}
            elif timed_out:
                run.process.kill()
                run.process.wait()
                rec["error"] = f"DeCART timeout after {timeout_s:g}s"
            else:
                still.append((rec, run))
        active = still


def _write_verdict(result: GeomValidationResult, scratch: Path,
                   log: Callable[[str], None]) -> None:
    """Write ``verdict.json`` + ``verdict.md`` to the scratch dir."""
    vj = scratch / "verdict.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": result.dry_run,
        "overall_verdict": result.overall_verdict,
        "n_requested": result.n_variants,
        "n_admissible": result.n_admissible,
        "pitch_ceiling_frac": PITCH_CEIL_FRAC,
        "acceptance_bands": {k: {"min_spearman": v[0], "max_mae": v[1]}
                             for k, v in ACCEPTANCE_BANDS.items()},
        "constraint_note": result.constraint_note,
        "variants": result.variants,
        "ood_warnings": result.ood_warnings,
    }
    vj.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    result.verdict_path = vj

    lines: list[str] = []
    lines.append("# Geometry-validation verdict (review sec. 4c)")
    lines.append("")
    lines.append(f"- mode: **{'DRY-RUN (no DeCART / no MASTER)' if result.dry_run else 'LIVE'}**")
    lines.append(f"- overall verdict: **{result.overall_verdict}**")
    lines.append(f"- variants: {result.n_admissible} admissible / {result.n_variants} requested "
                 f"(pin-pitch ceiling +{PITCH_CEIL_FRAC*100:.2f}%)")
    lines.append(f"- acceptance bands: " + "; ".join(
        f"{k} Sp>={v[0]:.2f} & MAE<={v[1]:g}" for k, v in ACCEPTANCE_BANDS.items()))
    lines.append("")
    lines.append("> " + result.constraint_note)
    lines.append("")
    lines.append("| variant | pitch | radius | verdict | f_r Sp/MAE | cyclen Sp/MAE | cbc Sp/MAE | OOD |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in result.variants:
        if v.get("verdict") == "ERROR":
            lines.append(f"| {v['type_id']} | — | — | ERROR | {v.get('error','')} | | | |")
            continue
        pt = v.get("per_target", {})
        def _cell(name: str) -> str:
            s = pt.get(name, {})
            return f"{_fmt(s.get('spearman'))}/{_fmt(s.get('mae'))}"
        ood = "yes" if v.get("ood_channels") else "no"
        lines.append(
            f"| {v['type_id']} | {v['pitch_frac']*100:+.1f}% | {v['radius_frac']*100:+.1f}% | "
            f"**{v.get('verdict','?')}** | {_cell('f_r')} | {_cell('cyclen')} | "
            f"{_cell('cbc_max')} | {ood} |")
    lines.append("")
    if result.ood_warnings:
        lines.append("## Serve-time OOD guard (sec. 4b) — flagged variants")
        lines.append("")
        lines.append("> Every geometry variant is expected to be OOD vs the fixed-geometry "
                     "training population (that is the whole point of the guard). A variant "
                     "that clears the transfer bands AND is OOD may enter the optimizer only "
                     "with the guard active; a variant that FAILS the bands quarantines the "
                     "geometry axis until a geometry-aware v5 retrain.")
        lines.append("")
        for w in result.ood_warnings:
            lines.append(f"- {w}")
        lines.append("")
    if result.dry_run:
        lines.append("_DRY-RUN scope: DeCART/MASTER were NOT run; physics seeded from anchors, "
                     "champion is the deterministic stub, truth is the StubEvaluator. This "
                     "validates the pipeline, not real transfer — run without --dry-run (DeCART "
                     "cap 4 + the standard MASTER verifier) for a licensing-grade verdict._")
        lines.append("")
    (scratch / "verdict.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"[geomcheck] verdict {result.overall_verdict} -> {scratch / 'verdict.md'}")


__all__ = [
    "ACCEPTANCE_BANDS",
    "DEFAULT_SCRATCH",
    "GeomCheckError",
    "GeomValidationResult",
    "GeomVariant",
    "PITCH_CEIL_FRAC",
    "StubProbeModel",
    "build_probe_patterns",
    "generate_variant_decks",
    "geom_variant_grid",
    "run_geom_validation",
    "run_probe_chains",
    "score_variant",
    "side_table_row",
    "side_table_vec",
    "write_variant_deck",
]
