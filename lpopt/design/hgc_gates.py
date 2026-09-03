"""HGC product gates G-H1 / G-H1b / G-H1c / G-H2 / G-H4 (assembly on-demand §6.4).

A DeCART2D FA product (``FA_<alias>.HGC``) is the *only* artefact that feeds the
MASTER library build, so it is gated before it is allowed anywhere near
``MAS_XSL`` / ``MAS_HFF``.  Every gate here is a **pure function over the parsed
HGC text** (plus, for G-H1b, the on-disk byte size) — no DeCART, no MASTER, no
pandas — so the whole set runs on a fixture in milliseconds.

The gates (``assembly_on_demand_design_v2_20260903.md`` §6.4, task list §4 #11):

``G-H1``  *structure*
    ``%TITL`` census is **334** = DEPL 62 + BRANCH 16x17, the four per-state tags
    ``%DIST`` / ``%MACX`` / ``%MICX`` / ``%ADFT`` appear 334 times each, the
    ``CASE ::`` census is REFERENCE 62 + 16 distinct branch labels x 17, and the
    file closes with exactly one ``%FINE``.  A missing branch is never tolerated:
    ``CRD1*`` only exists in the CR1 branches and the MASTER deck runs
    ``%JOB_MDL irod=2`` — whether MASTER dies or silently mis-computes on a
    truncated roster is **unverified** (risk R11), so the policy is *no omission*.

``G-H1b`` *size*
    Every produced APR1400 FA HGC with ``n_gd in {12, 16, 20, 24}`` is **exactly
    7,395,955 B**.  Any other ``n_gd`` is **ABSTAIN, not FAIL**: the 6,867,567 B
    of the ``n_gd = 0`` V01/V02 pair is un-explained (open question 21) and
    ``n_gd in {4, 8}`` has never been produced, so there is no measured constant
    to gate against.

``G-H1c`` *validity*
    The structural liveness check of ``lattice._hgc_looks_valid`` (readable,
    non-trivial, carries ``%TITL`` + ``CASE ::`` + ``%DIST``) **plus** a per-BU
    sanity pass on the reference depletion curve: burnups strictly increasing,
    and k-inf monotonically decreasing after the Gd-burnout peak (a rise of more
    than :data:`K_MONOTONE_TOL` beyond the peak is a corrupt/interleaved product,
    not physics).  The Gd-burnout peak is the **last significant k-inf local
    maximum at BU <= 30 MWd/kg** (:func:`burnout_peak_index`), *not* the global
    maximum — see the AMENDMENT note there.

``G-H2``  *Gd census*
    The BOC pin-power census (``0 < power < 0.6``, the rule validated in
    ``lpopt/data/fuel_types.py:554`` ``count_gd_pins_from_hgc`` against
    ``0_APR1400/260624/hgc/FA_B1.HGC``) equals the **requested** ``n_gd`` — 20/20
    for the slice-Z designs.  ``count_gd_pins_from_text`` here is the text-domain
    twin of that function; ``tests/test_hgc_gates.py`` cross-checks the two agree.

``G-H4``  *screen regression*
    Over the whole ``BU >= 0.2`` range, ``|rho_(A) - rho_DeCART| <= 350 pcm``
    (``rho = 1 - 1/k``) and ``|FF_(A) - FF_DeCART| <= 0.0021``.  **This is a
    regression check, not a first measurement** — both thresholds are the
    measured maxima of the T3-T6 holdout, so the surrogate's density/Xe
    -convention bias is already bounded (risk R4).  A failure means the screen no
    longer predicts the real calculation, and the on-demand pipeline stops there.
    The k half was **amended on 2026-09-03** (quantity ``rho`` not ``k``; bar 350
    not 100) after the holdout was re-measured — see
    :func:`gate_h4_screen_regression` for the derivation and the evidence.

Nothing in this module reads a config or touches the filesystem except the thin
``*_for_file`` convenience wrappers at the end.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# registered constants (design v2 §5.2 / §6.4)
# --------------------------------------------------------------------------- #
#: DEPL points in the frozen golden deck (``burnup 0.0 0.2 0.5 1 -45/1.0 -80/2.5``).
N_DEPL_STATES = 62
#: BRANCH cases: 1 BORON / 2 TFUEL / 3-8 DMOD1..6 / 9 CR1 REF / 10 CR1 BOR /
#: 11-16 CR1 DMOD1..6.
N_BRANCHES = 16
#: Common branch burnup grid ``0 0.2 0.5 1 3 5 7 10 15 20 25 30 40 50 60 70 80``.
N_BRANCH_POINTS = 17
#: ``62 + 16*17`` — the ``%TITL`` census, and the count of every per-state tag.
N_TITL_EXPECTED = N_DEPL_STATES + N_BRANCHES * N_BRANCH_POINTS      # 334

#: Per-state tags that must appear exactly ``N_TITL_EXPECTED`` times.
STATE_TAGS = ("%DIST", "%MACX", "%MICX", "%ADFT")
#: Reference (depletion) case label.
CASE_REFERENCE = "REFERENCE CASE"

#: Exact byte size of every produced APR1400 FA HGC whose ``n_gd`` is gateable.
HGC_SIZE_BYTES = 7_395_955
#: ``n_gd`` values for which :data:`HGC_SIZE_BYTES` is a measured constant.
HGC_SIZE_GATED_N_GD = (12, 16, 20, 24)
#: Observed size of the ``n_gd = 0`` V01/V02 products — mechanism UNKNOWN, so it
#: is documented, never gated (open question 21).
HGC_SIZE_BYTES_NO_GD = 6_867_567

#: A pin whose BOC relative power is in ``(0, GD_POWER_MAX)`` carries Gd; a guide
#: tube / instrument position reads exactly ``0.000``.
GD_POWER_MAX = 0.6
#: Pin-map row width of the ``%DIST`` block (16x16 assembly).
DIST_ROW_WIDTH = 16

#: G-H1c: a post-peak k-inf *rise* larger than this (in k, i.e. 100 pcm) is a
#: corrupt product rather than Gd-burnout physics.  It is also the minimum rise
#: (above the preceding trough) that makes a local maximum count as *the*
#: Gd-burnout peak rather than numerical ripple.
K_MONOTONE_TOL = 1.0e-3
#: G-H1c: the Gd-burnout peak is searched for at or below this burnup.  Measured
#: over all 37 approved library products (S4-B, 2026-09-03) the latest burnout
#: peak is **Q3 at BU = 25.0** (n_gd = 24), so a 25.0 window would carry zero
#: margin on a product already in the library; 30.0 carries 5 MWd/kg.  A k-inf
#: rise *after* this window is never Gd physics, and that is what G-H1c polices;
#: inside it, the mandatory burnout rise makes monotonicity meaningless.
GD_BURNOUT_BU_MAX = 30.0
#: G-H4 threshold — AMENDMENT 2026-09-03 (S4-B), see :func:`gate_h4_screen_regression`.
#: The registered quantity is |Delta rho| in pcm (``rho = 1 - 1/k``), the
#: quantity ``opmodel/s02_surrogate_vs_decart.py`` actually differenced; the bar
#: is the T3-T6 holdout maximum 303.8 pcm plus ~15 % margin, rounded up to 50.
#: The superseded 100.0 was a mis-reading of OPSCREEN.md:169 (see amendment).
G_H4_K_TOL_PCM = 350.0
#: Superseded G-H4 k bar, kept for provenance only — never gate against it.
G_H4_K_TOL_PCM_SUPERSEDED = 100.0
G_H4_FF_TOL = 0.0021
#: G-H4 is evaluated over ``BU >= 0.2`` GWd/tU (the fresh point is excluded).
G_H4_BU_MIN = 0.2
#: Burnup grid points are matched between screen and HGC within this window.
BU_MATCH_TOL = 1.0e-3

PASS = "PASS"
FAIL = "FAIL"
ABSTAIN = "ABSTAIN"

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


class HgcGateError(ValueError):
    """Raised for an argument the gates cannot interpret (never for a FAIL)."""


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateResult:
    """One gate's verdict.

    ``status`` is :data:`PASS`, :data:`FAIL`, or :data:`ABSTAIN` — ABSTAIN means
    *there is no registered expectation for this input*, which is a hand-off to
    manual review, **not** a pass and **not** a failure.
    """

    gate: str
    status: str
    detail: str
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True only for :data:`PASS` (an ABSTAIN is not a pass)."""
        return self.status == PASS

    def as_dict(self) -> dict:
        return {"gate": self.gate, "status": self.status,
                "detail": self.detail, "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class RefPoint:
    """One reference-case depletion state: burnup [GWd/tU], k-inf, peak pin FF."""

    burnup: float
    kinf: float
    ff_pin_max: float | None = None


def verdict(results: Iterable[GateResult]) -> str:
    """Fold gate results: any FAIL -> FAIL, else any ABSTAIN -> ABSTAIN, else PASS."""
    statuses = [r.status for r in results]
    if not statuses:
        return ABSTAIN
    if FAIL in statuses:
        return FAIL
    if ABSTAIN in statuses:
        return ABSTAIN
    return PASS


# --------------------------------------------------------------------------- #
# parsing (pure, over the HGC text)
# --------------------------------------------------------------------------- #
def _floats(line: str) -> list[float]:
    out: list[float] = []
    for tok in _FLOAT_RE.findall(line):
        try:
            out.append(float(tok.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    return out


def iter_state_blocks(text: str) -> Iterable[list[str]]:
    """Yield the line list of every ``%TITL`` state block (leading ``%TITL`` dropped)."""
    for blk in re.split(r"(?m)^%TITL", text):
        if blk.strip():
            yield blk.splitlines()


def _case_label(block_lines: Sequence[str]) -> tuple[str, int] | None:
    for i, ln in enumerate(block_lines):
        if "CASE ::" in ln:
            return ln.split("CASE ::", 1)[1].strip(), i
    return None


def _dist_map_flat(block_lines: Sequence[str]) -> list[float]:
    """First ``%DIST`` 16x16 map of a block, flattened; ``[]`` when absent.

    Mirrors ``fuel_types._first_dist_map_flat``: only exact-16-value rows are
    accepted, so the scan stops cleanly at the next ``%`` block or a short row.
    """
    di = None
    for i, ln in enumerate(block_lines):
        if ln.strip().startswith("%DIST"):
            di = i
            break
    if di is None:
        return []
    rows: list[list[float]] = []
    for ln in block_lines[di + 1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("%"):
            break
        try:
            vals = [float(t) for t in s.split()]
        except ValueError:
            break
        if len(vals) != DIST_ROW_WIDTH:
            break
        rows.append(vals)
        if len(rows) >= DIST_ROW_WIDTH:
            break
    if len(rows) < DIST_ROW_WIDTH:
        return []
    return [v for row in rows[:DIST_ROW_WIDTH] for v in row]


def parse_structure(text: str) -> dict:
    """Census of the structural markers G-H1 gates.

    Returns ``{"titl": int, "fine": int, "tags": {tag: count},
    "cases": {label: count}}``.
    """
    # Anchored at line start: a tag is a block marker, never inline data.
    tags = {tag: len(re.findall(r"(?m)^" + re.escape(tag), text)) for tag in STATE_TAGS}
    cases: dict[str, int] = {}
    titl = 0
    for block_lines in iter_state_blocks(text):
        titl += 1
        found = _case_label(block_lines)
        if found is not None:
            cases[found[0]] = cases.get(found[0], 0) + 1
    return {
        "titl": titl,
        "fine": len(re.findall(r"(?m)^%FINE", text)),
        "tags": tags,
        "cases": cases,
    }


def reference_curve(text: str) -> list[RefPoint]:
    """The reference (depletion) case curve: burnup, k-inf, peak pin power.

    Block layout after ``CASE ::`` (shared with ``fuel_types._parse_hgc_block_state``):
    an integer descriptor line, then ``sp_power BURNUP kinf kcrit Bsq Tfuel`` and
    ``Tcool ppm _ pressure rho_cool pnorm``.
    """
    points: list[RefPoint] = []
    for block_lines in iter_state_blocks(text):
        found = _case_label(block_lines)
        if found is None or found[0] != CASE_REFERENCE:
            continue
        _label, cidx = found
        nums: list[list[float]] = []
        for ln in block_lines[cidx + 1:cidx + 7]:
            vals = _floats(ln)
            if vals:
                nums.append(vals)
            if len(nums) >= 3:
                break
        if len(nums) < 3 or len(nums[1]) < 6:
            continue
        flat = _dist_map_flat(block_lines)
        points.append(RefPoint(burnup=nums[1][1], kinf=nums[1][2],
                               ff_pin_max=(max(flat) if flat else None)))
    return points


def count_gd_pins_from_text(text: str) -> int:
    """Gd-pin census from the first (BOC) ``%DIST`` map — text-domain twin of
    ``lpopt.data.fuel_types.count_gd_pins_from_hgc`` (same ``0 < p < 0.6`` rule,
    same first-map scope); the two are cross-checked in the gate tests."""
    for block_lines in iter_state_blocks(text):
        flat = _dist_map_flat(block_lines)
        if flat:
            return sum(1 for v in flat if 0.0 < v < GD_POWER_MAX)
    return 0


# --------------------------------------------------------------------------- #
# G-H1 — structure
# --------------------------------------------------------------------------- #
def gate_h1_structure(text: str) -> GateResult:
    """``%TITL`` 334 = 62 + 16x17, four tags x 334, CASE census, one ``%FINE``."""
    st = parse_structure(text)
    problems: list[str] = []

    if st["titl"] != N_TITL_EXPECTED:
        problems.append(
            f"%TITL {st['titl']} != {N_TITL_EXPECTED} "
            f"(= DEPL {N_DEPL_STATES} + BRANCH {N_BRANCHES}x{N_BRANCH_POINTS})")
    for tag in STATE_TAGS:
        if st["tags"][tag] != N_TITL_EXPECTED:
            problems.append(f"{tag} {st['tags'][tag]} != {N_TITL_EXPECTED}")
    if st["fine"] != 1:
        problems.append(f"%FINE {st['fine']} != 1")

    cases = dict(st["cases"])
    n_ref = cases.pop(CASE_REFERENCE, 0)
    if n_ref != N_DEPL_STATES:
        problems.append(f"'{CASE_REFERENCE}' {n_ref} != {N_DEPL_STATES}")
    if len(cases) != N_BRANCHES:
        problems.append(f"branch labels {len(cases)} != {N_BRANCHES} "
                        f"(missing branches are never tolerated: risk R11)")
    bad = {k: v for k, v in cases.items() if v != N_BRANCH_POINTS}
    if bad:
        problems.append(f"branch label counts != {N_BRANCH_POINTS}: {sorted(bad.items())}")

    metrics = {"titl": st["titl"], "fine": st["fine"], "tags": st["tags"],
               "n_reference": n_ref, "n_branch_labels": len(cases)}
    if problems:
        return GateResult("G-H1", FAIL, "; ".join(problems), metrics)
    return GateResult(
        "G-H1", PASS,
        f"{N_TITL_EXPECTED} states, {N_BRANCHES} branches x {N_BRANCH_POINTS}, "
        f"{N_DEPL_STATES} reference points, all tags complete", metrics)


# --------------------------------------------------------------------------- #
# G-H1b — size (ABSTAIN outside the measured n_gd set)
# --------------------------------------------------------------------------- #
def gate_h1b_size(size_bytes: int, n_gd: int) -> GateResult:
    """Exact 7,395,955 B for ``n_gd in {12,16,20,24}``; **ABSTAIN** otherwise."""
    metrics = {"size_bytes": int(size_bytes), "n_gd": int(n_gd),
               "expected_bytes": HGC_SIZE_BYTES}
    if int(n_gd) not in HGC_SIZE_GATED_N_GD:
        metrics["expected_bytes"] = None
        note = ""
        if int(n_gd) == 0:
            note = (f" (the n_gd=0 products measure {HGC_SIZE_BYTES_NO_GD} B; "
                    f"the mechanism is unexplained — open question 21)")
        return GateResult(
            "G-H1b", ABSTAIN,
            f"n_gd {n_gd} is outside the measured set {list(HGC_SIZE_GATED_N_GD)}: "
            f"no registered size constant, hand to manual review{note}", metrics)
    if int(size_bytes) != HGC_SIZE_BYTES:
        return GateResult(
            "G-H1b", FAIL,
            f"size {int(size_bytes)} B != {HGC_SIZE_BYTES} B for n_gd={n_gd}",
            metrics)
    return GateResult("G-H1b", PASS,
                      f"size {HGC_SIZE_BYTES} B exact for n_gd={n_gd}", metrics)


# --------------------------------------------------------------------------- #
# G-H1c — validity + per-BU sanity
# --------------------------------------------------------------------------- #
def burnout_peak_index(burnups: Sequence[float], kinfs: Sequence[float], *,
                       window: float = GD_BURNOUT_BU_MAX,
                       tol: float = K_MONOTONE_TOL) -> int:
    """Index of the **Gd-burnout peak** on a reference depletion curve.

    **Rule (G-H1c, amended 2026-09-03).**  The burnout peak is the *last* k-inf
    local maximum at ``burnup <= window`` whose rise above the preceding trough
    exceeds ``tol`` — equivalently, the first local maximum *after* the initial
    Gd-dominated segment.  Index 0 (BOC) is the fallback when the curve carries
    no such hump, which is the correct answer for an un-poisoned lattice.

    **Why not the global maximum.**  A high-Gd assembly has its *global* k-inf
    maximum at BOC, and the Gd-burnout rise that follows the initial dip is
    physically mandatory (BU ~ 9-21 for n_gd = 20).  Reading the global maximum
    as the burnout peak therefore condemns that mandatory rise as non-monotone:
    the superseded logic failed 5 of the 6 n_gd = 20 HGCs already approved into
    the library (P2, P3, P9, Q4, T1), and passed the sixth (S9) only because its
    BOC k-inf happened to sit *below* its burnout peak.  See the S4-B stamp in
    ``assembly_slice_Z_prereg_20260903_DRAFT.md``.
    """
    n = len(kinfs)
    if n == 0:
        return 0
    peak = 0
    trough = float(kinfs[0])
    for i in range(1, n):
        if float(burnups[i]) > window:
            break
        k = float(kinfs[i])
        if k < trough:
            trough = k
        elif (k > float(kinfs[i - 1])
              and (i + 1 >= n or k >= float(kinfs[i + 1]))
              and k - trough > tol):
            peak = i
            trough = k
    return peak


def gate_h1c_validity(text: str, *, size_bytes: int | None = None) -> GateResult:
    """``_hgc_looks_valid`` liveness + monotonic sanity of the reference curve.

    The monotone requirement applies **only after the Gd-burnout peak** as
    defined by :func:`burnout_peak_index` (the last significant k-inf local
    maximum at ``BU <= GD_BURNOUT_BU_MAX``), never after the global maximum.
    """
    problems: list[str] = []
    if size_bytes is not None and int(size_bytes) < 256:
        problems.append(f"size {int(size_bytes)} B < 256 B (truncated product)")
    for marker in ("%TITL", "CASE ::", "%DIST"):
        if marker not in text:
            problems.append(f"missing {marker!r}")

    points = reference_curve(text)
    metrics: dict = {"n_reference_points": len(points)}
    if not points:
        problems.append("no reference-case state block parsed")
        return GateResult("G-H1c", FAIL, "; ".join(problems), metrics)

    bus = [p.burnup for p in points]
    kinfs = [p.kinf for p in points]
    metrics.update(burnup_min=min(bus), burnup_max=max(bus),
                   kinf_first=kinfs[0], kinf_last=kinfs[-1])

    non_increasing = [(bus[i - 1], bus[i]) for i in range(1, len(bus))
                      if bus[i] <= bus[i - 1]]
    if non_increasing:
        problems.append(f"burnup axis not strictly increasing at {non_increasing[:3]}")

    peak = burnout_peak_index(bus, kinfs)
    metrics["kinf_peak_burnup"] = bus[peak]
    metrics["kinf_peak_index"] = peak
    metrics["kinf_global_max_burnup"] = bus[max(range(len(kinfs)),
                                               key=lambda i: kinfs[i])]
    metrics["burnout_window_bu"] = float(GD_BURNOUT_BU_MAX)
    rises = [(bus[i], kinfs[i] - kinfs[i - 1])
             for i in range(peak + 1, len(kinfs))
             if kinfs[i] - kinfs[i - 1] > K_MONOTONE_TOL]
    if rises:
        problems.append(
            f"k-inf rises by > {K_MONOTONE_TOL:g} after the Gd-burnout peak at "
            f"BU={bus[peak]:g}: {[(b, round(d, 6)) for b, d in rises[:3]]}")

    if problems:
        return GateResult("G-H1c", FAIL, "; ".join(problems), metrics)
    return GateResult(
        "G-H1c", PASS,
        f"structurally live; {len(points)} reference points, BU {min(bus):g}-"
        f"{max(bus):g} strictly increasing, k-inf monotone after the Gd-burnout "
        f"peak at BU={bus[peak]:g}", metrics)


# --------------------------------------------------------------------------- #
# G-H2 — Gd census
# --------------------------------------------------------------------------- #
def gate_h2_gd_census(text: str, n_gd: int) -> GateResult:
    """BOC pin-power Gd census == the **requested** ``n_gd`` (20/20 for slice Z)."""
    counted = count_gd_pins_from_text(text)
    metrics = {"counted": counted, "requested": int(n_gd)}
    if counted != int(n_gd):
        return GateResult("G-H2", FAIL,
                          f"Gd census {counted} != requested n_gd {int(n_gd)}",
                          metrics)
    return GateResult("G-H2", PASS,
                      f"Gd census {counted}/{int(n_gd)}", metrics)


# --------------------------------------------------------------------------- #
# G-H4 — surrogate screen regression
# --------------------------------------------------------------------------- #
def _as_series(series: Mapping[float, float] | Sequence[tuple[float, float]] | None
               ) -> list[tuple[float, float]]:
    if series is None:
        return []
    if isinstance(series, Mapping):
        items = list(series.items())
    else:
        items = [tuple(pair) for pair in series]
    out: list[tuple[float, float]] = []
    for pair in items:
        if len(pair) != 2:
            raise HgcGateError(f"screen series entry {pair!r} is not (burnup, value)")
        out.append((float(pair[0]), float(pair[1])))
    return sorted(out)


def _rho(k: float) -> float:
    """Reactivity ``1 - 1/k`` — the quantity OPSCREEN differences in pcm."""
    k = float(k)
    if k == 0.0:
        raise HgcGateError("k = 0 has no reactivity")
    return 1.0 - 1.0 / k


def _match(bu: float, points: Sequence[tuple[float, float]]) -> float | None:
    for b, v in points:
        if abs(b - bu) <= BU_MATCH_TOL:
            return v
    return None


def gate_h4_screen_regression(
    hgc_points: Sequence[RefPoint],
    screen_kinf: Mapping[float, float] | Sequence[tuple[float, float]],
    screen_ff: Mapping[float, float] | Sequence[tuple[float, float]] | None = None,
    *,
    k_tol_pcm: float = G_H4_K_TOL_PCM,
    ff_tol: float = G_H4_FF_TOL,
    bu_min: float = G_H4_BU_MIN,
) -> GateResult:
    """``|rho_(A) - rho_DeCART| <= 350 pcm`` and ``|FF_(A) - FF_DeCART| <= 0.0021``
    over every ``BU >= bu_min`` point the screen and the HGC share.

    A **regression** check: both thresholds are the T3-T6 holdout measured maxima
    plus margin.  With no shared burnup point the verdict is ABSTAIN — an empty
    comparison is not evidence of agreement.

    **AMENDMENT 2026-09-03 (S4-B).**  Two registered corrections, both evidenced
    in the S4-B stamp of ``assembly_slice_Z_prereg_20260903_DRAFT.md``:

    1. *Quantity.*  The gate differenced ``k`` and scaled by 1e5.  The quantity
       OPSCREEN actually calibrated on is **reactivity**,
       ``rho = 1 - 1/k`` (``opmodel/s02_surrogate_vs_decart.py``:
       ``rs, rd = 1 - 1/s, 1 - 1/d ... (rs - rd) * 1e5``), which is also what the
       operating-point model consumes.  ``|Delta k| * 1e5`` is not "pcm" and is
       ~1/k**2 off it — 21 % high at k = 1.1, 66 % low at k = 0.77.
    2. *Bar.*  ``100.0`` was a mis-reading of ``OPSCREEN.md:169``.  Re-run on its
       own defining holdout with the correct quantity, the *stored* DeCART truth
       and the *same* surrogate checkpoint, T3-T6 measure
       **157.1 / 303.8 / 120.7 / 115.3 pcm** (maxima at BU 15/19/17/17 — inside
       the Gd-burnout shoulder, where ``s02``'s 18-point display grid
       ``[.. 12, 15, 20, 25 ..]`` has no sample point, which is how the coarse
       table read "< 100").  The wiring was verified exact rather than assumed:
       HGC ``%TITL`` reference k-inf equals the ``.out`` ``K-CONV`` series to
       <= 0.5 pcm, both BU grids are the identical 62 points the surrogate
       returns, and this harness reproduces OPSCREEN's recorded FF for all four
       holdout members to 4 decimals (1.1073 / 1.1409 / 1.1012 / 1.1011
       surrogate, 1.1090 / 1.1430 / 1.1020 / 1.1020 DeCART) and its -2200 pcm
       BU = 0 Xe artefact.  Nothing was left to fix in the wiring, so the bar is
       re-derived from the holdout: ``max 303.8 * 1.15 -> 350`` pcm.

    The FF half needed no amendment: it reproduced its 0.0021 bar as recorded.
    """
    kinf_screen = _as_series(screen_kinf)
    ff_screen = _as_series(screen_ff)

    dk: list[tuple[float, float]] = []          # (burnup, pcm)
    dff: list[tuple[float, float]] = []
    for p in hgc_points:
        if p.burnup + BU_MATCH_TOL < bu_min:
            continue
        k_a = _match(p.burnup, kinf_screen)
        if k_a is not None:
            dk.append((p.burnup, abs(_rho(k_a) - _rho(p.kinf)) * 1.0e5))
        if ff_screen and p.ff_pin_max is not None:
            ff_a = _match(p.burnup, ff_screen)
            if ff_a is not None:
                dff.append((p.burnup, abs(ff_a - p.ff_pin_max)))

    metrics: dict = {"n_kinf_compared": len(dk), "n_ff_compared": len(dff),
                     "k_tol_pcm": float(k_tol_pcm), "ff_tol": float(ff_tol),
                     "bu_min": float(bu_min), "k_metric": "abs_drho_pcm"}
    if not dk and not dff:
        return GateResult("G-H4", ABSTAIN,
                          f"no screen/HGC point shares a burnup at BU >= {bu_min:g}",
                          metrics)

    problems: list[str] = []
    if dk:
        bu_k, worst_k = max(dk, key=lambda t: t[1])
        metrics.update(max_drho_pcm=worst_k, max_drho_burnup=bu_k,
                       max_dk_pcm=worst_k, max_dk_burnup=bu_k)
        if worst_k > float(k_tol_pcm):
            problems.append(f"max |drho| {worst_k:.1f} pcm > {float(k_tol_pcm):g} pcm "
                            f"at BU={bu_k:g}")
    if dff:
        bu_f, worst_f = max(dff, key=lambda t: t[1])
        metrics.update(max_dff=worst_f, max_dff_burnup=bu_f)
        if worst_f > float(ff_tol):
            problems.append(f"max |dFF| {worst_f:.4f} > {float(ff_tol):g} at BU={bu_f:g}")

    if problems:
        return GateResult("G-H4", FAIL, "; ".join(problems), metrics)
    detail = f"{len(dk)} k point(s)"
    if dk:
        detail += f", max |drho| {metrics['max_drho_pcm']:.1f} pcm"
    if dff:
        detail += f"; {len(dff)} FF point(s), max |dFF| {metrics['max_dff']:.4f}"
    return GateResult("G-H4", PASS, detail + " — screen still predicts DeCART", metrics)


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def run_gates(
    text: str,
    *,
    n_gd: int,
    size_bytes: int | None = None,
    screen_kinf: Mapping[float, float] | Sequence[tuple[float, float]] | None = None,
    screen_ff: Mapping[float, float] | Sequence[tuple[float, float]] | None = None,
) -> list[GateResult]:
    """G-H1, G-H1b, G-H1c, G-H2 and (when a screen series is given) G-H4.

    ``size_bytes = None`` skips G-H1b (nothing to gate); an omitted screen series
    skips G-H4 rather than passing it vacuously.
    """
    results = [gate_h1_structure(text)]
    if size_bytes is not None:
        results.append(gate_h1b_size(size_bytes, n_gd))
    results.append(gate_h1c_validity(text, size_bytes=size_bytes))
    results.append(gate_h2_gd_census(text, n_gd))
    if screen_kinf is not None or screen_ff is not None:
        results.append(gate_h4_screen_regression(
            reference_curve(text), screen_kinf or (), screen_ff))
    return results


def run_gates_for_file(
    hgc_path: str | Path,
    *,
    n_gd: int,
    screen_kinf: Mapping[float, float] | Sequence[tuple[float, float]] | None = None,
    screen_ff: Mapping[float, float] | Sequence[tuple[float, float]] | None = None,
) -> list[GateResult]:
    """:func:`run_gates` on a product file (the only filesystem-touching entry)."""
    path = Path(hgc_path)
    if not path.is_file():
        return [GateResult("G-H1c", FAIL, f"no HGC product at {path}",
                           {"path": str(path)})]
    size_bytes = path.stat().st_size
    with open(path, "r", errors="replace") as handle:
        text = handle.read()
    return run_gates(text, n_gd=n_gd, size_bytes=size_bytes,
                     screen_kinf=screen_kinf, screen_ff=screen_ff)


__all__ = [
    "ABSTAIN",
    "FAIL",
    "GD_BURNOUT_BU_MAX",
    "GD_POWER_MAX",
    "G_H4_FF_TOL",
    "G_H4_K_TOL_PCM",
    "G_H4_K_TOL_PCM_SUPERSEDED",
    "GateResult",
    "HGC_SIZE_BYTES",
    "HGC_SIZE_BYTES_NO_GD",
    "HGC_SIZE_GATED_N_GD",
    "HgcGateError",
    "N_BRANCHES",
    "N_BRANCH_POINTS",
    "N_DEPL_STATES",
    "K_MONOTONE_TOL",
    "N_TITL_EXPECTED",
    "PASS",
    "RefPoint",
    "burnout_peak_index",
    "count_gd_pins_from_text",
    "gate_h1_structure",
    "gate_h1b_size",
    "gate_h1c_validity",
    "gate_h2_gd_census",
    "gate_h4_screen_regression",
    "iter_state_blocks",
    "parse_structure",
    "reference_curve",
    "run_gates",
    "run_gates_for_file",
    "verdict",
]
