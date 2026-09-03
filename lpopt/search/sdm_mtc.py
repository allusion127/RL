"""SDM / MTC post-verification harness (plan section 12.5).

MTC (moderator temperature coefficient) and SDM (shutdown margin) are **user-set
design/licensing constraints** — they are *not* model-predicted figures of merit.
This module enforces them as a **post-verification stage** on top-K converged,
spec-feasible candidates of a campaign run: for each candidate it synthesises a
small number of MASTER *branch* decks from the candidate's own final-cycle deck +
its converged restart and runs them (≈2 extra MASTER calls per candidate, each a
single steady-state / branch set of ~20-60 s — *no* equilibrium depletion chain).

Ported from the MOCHA post-SA licensing harness
``...\\2_LP\\MOCHA\\sdm_mtc*.py`` (read 2026-07-17):

* ``sdm_mtc_io.py`` — the ``%EXE_RHO`` / ``%EXE_ROD`` / ``%ROD_CFG`` / ``%ROD_MAP``
  card idioms (``build_mtc_branch_full`` L400-419, ``_exe_rho_mtc`` L421-432,
  ``build_sdm_branch`` L361-397, ``rod_cfg_block`` L292-299, ``rod_map_block``
  L302-311, ``_insert_lines`` L338-358);
* ``sdm_mtc_parse.py`` — reactivity arithmetic (``rho_pcm`` L12-13,
  ``mtc_two_point_from_rows`` L77-91, ``parse_mtc_from_out`` L40-57,
  ``mtc_text_to_pcm_per_c`` L60-68);
* ``sdm_mtc.py`` — the per-case orchestration (``_run_mtc`` L256-308, ``_run_sdm``
  L311-379) and pass/fail semantics;
* ``sdm_mtc_types.py`` — the typed records + DCD limits (``LicensingLimits``
  L9-25, ``CaseLicensingResult.pass_limit`` L84-95).

**Audit-corrected MTC scaling (2026-06 MOCHA adversarial audit, finding F1).**
The pre-audit parser applied a *magnitude-based* rescale
``if abs(value) < 10.0: return value * 10.0`` — which turned a physical
+5 pcm/°C into +50 pcm/°C and produced a **false licensing FAIL** against the
``MTC ≤ +9 pcm/°C`` window, while the keff two-point fallback (which never went
through the rescale) disagreed by 10×.  The corrected MOCHA source (verified
2026-07-17, ``sdm_mtc_parse.py`` L40-68) removes the heuristic entirely:
``parse_mtc_from_out`` returns the **raw** printed value and unit conversion is
**explicit and deterministic** in :func:`mtc_text_to_pcm_per_c`
(``pcm_per_c`` → identity, ``drho_per_c_1e-4`` → ×10, i.e. 1e-4 Δρ/°C → pcm/°C).
This port carries the corrected form and additionally uses the unit-safe keff
**two-point central difference as the PRIMARY** MTC estimator (audit F1/F8
recommendation), with the text parse as a cross-check/fallback only.

Sign / unit conventions (mirrored from MOCHA + APR1400 DCD Table 4.3):

* **MTC** in **pcm/°C** (``ρ = (k-1)/k · 1e5``); the acceptance window is
  ``mtc_min ≤ value ≤ mtc_max`` with ``mtc_max`` the **most-positive allowed**
  MTC (DCD-constant default window ``[-54, +9] pcm/°C``).  A physically negative
  MTC at power passes; a positive MTC above +9 fails.
* **SDM** in **pcm**; ``required = cea_allowance + net_worth_uncertainty``
  (DCD default ``10180 + 690 = 10870 pcm``).  ``available = W_ari − worst_stuck``
  (net rod worth with the single most-reactive rod stuck fully withdrawn);
  ``margin = available − required``; **pass iff margin ≥ 0**.  ``limit = minimum
  required``.

Results are appended to a **sidecar** store ``data/sdm_mtc/results.jsonl`` keyed
by ``record_id`` (NOT the main store), so accumulated licensing labels can later
justify promoting MTC/SDM to predicted targets (plan note).

Decision D9 (2026-07-25) — the PRE-DELIVERY GATE
------------------------------------------------
*"F_r 제외 feasible 노심에 대해서 (평탄도 높음) SDM, MTC 검증 실시."*  The
``flat_power`` objective removed F_r from the objective (it survives only as a
1.70 SAFETY gate), which is exactly what lets the search flatten — and flattening
**monotonically degrades control-rod worth and increases leakage**, which no
search axis measures.  :func:`post_verify_delivery` therefore runs this harness
on the campaign's delivery-ranked candidates (``delivery.json`` — flat-band
candidates that are feasible EXCLUDING F_r) and marks violators before hand-off.

Two accounting facts this module refuses to hide:

* **Extra MASTER calls are counted, not estimated.**  Every branch run increments
  :attr:`SdmMtcResult.master_calls`; the campaign adds them to
  ``post_verify_master_calls`` in ``status.json`` and reports search budget and
  licensing budget as two separate numbers that also sum to a stated total.
* **A candidate is verified against ITS OWN converged restart or not at all.**
  Pairing candidate A's deck with candidate B's restart yields a plausible number
  that is meaningless.  Resolution goes through the per-candidate provenance index
  (:func:`load_targets`, written by the campaign from the converged equilibrium
  chain); an unresolvable candidate is SKIPPED with a reason, never approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from .._proc import no_window_flags
from ..vendor.masterrl.equilibrium import advance_cycle_deck, deck_cycle
from ..vendor.masterrl.master import MasterRunError, MasterRunner

__all__ = [
    "DEFAULT_SIDECAR",
    "TARGETS_NAME",
    "BranchParams",
    "CandidateRef",
    "LicensingLimits",
    "MtcResult",
    "PostVerifySummary",
    "RodGroup",
    "RodModel",
    "RodWorth",
    "SdmResult",
    "SdmMtcResult",
    "BranchExecutor",
    "BranchOutputs",
    "VendorBranchExecutor",
    "append_sidecar",
    "build_branch_decks",
    "candidates_from_delivery",
    "load_sidecar",
    "load_targets",
    "mtc_text_to_pcm_per_c",
    "mtc_two_point_from_rows",
    "parse_branch_keffs",
    "parse_mtc_from_out",
    "post_verify_delivery",
    "post_verify_topk",
    "record_target",
    "rho_pcm",
    "run_post_verification",
    "sdm_branch_labels",
]

#: Default sidecar store (NOT the main data/store) — accumulated licensing labels.
DEFAULT_SIDECAR = Path("data/sdm_mtc/results.jsonl")

#: Per-candidate branch-asset provenance index, written by the campaign into its
#: own run dir: one JSON object per converged candidate carrying the deck and the
#: CONVERGED equilibrium restart that candidate produced.  Without an entry a
#: candidate cannot be licence-verified (see module docstring).
TARGETS_NAME = "sdm_mtc_targets.jsonl"


# --------------------------------------------------------------------------- #
# reactivity arithmetic (port of MOCHA/sdm_mtc_parse.py)
# --------------------------------------------------------------------------- #
def rho_pcm(k_eff: float) -> float:
    """Reactivity in pcm from a multiplication factor (MOCHA sdm_mtc_parse L12)."""

    k = float(k_eff)
    return ((k - 1.0) / k) * 1.0e5


def mtc_text_to_pcm_per_c(value: float, units: str) -> float:
    """Convert an explicitly unit-tagged MTC text value to pcm/°C.

    **Audit-corrected (F1).**  No magnitude heuristic — the unit is an explicit
    setting, so the conversion is deterministic:

    * ``pcm_per_c``        → identity;
    * ``drho_per_c_1e-4``  → ×10 (1e-4 Δρ/°C == 10 pcm/°C, since 1 pcm = 1e-5 Δρ).

    Mirrors MOCHA sdm_mtc_parse.py L60-68 (the corrected, post-audit form).
    """

    u = str(units).lower()
    if u == "pcm_per_c":
        return float(value)
    if u == "drho_per_c_1e-4":
        return float(value) * 10.0
    raise ValueError("MTC output units must be 'pcm_per_c' or 'drho_per_c_1e-4'")


# The %EXE_RHO card is echoed into MAS_OUT ("mtc   0"), so a bare "MTC <number>"
# would capture the echo and report a fake 0.0 pcm/°C (MOCHA audit T7-3).  The
# value pattern therefore REQUIRES the trailing unit token.  Verified
# master4.0m4_r1 result line:  " 00:10.61  ...  1.001165  MTC   -23.12 PCM/C".
_MTC_PATTERNS = (
    re.compile(r"\bMTC\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*PCM\s*/\s*C\b", re.I),
    re.compile(
        r"MODERATOR\s+TEMPERATURE\s+COEF(?:F|FICIENT)?\.?[^-+0-9\n]*"
        r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*PCM\s*/\s*C\b",
        re.I,
    ),
)


def parse_mtc_from_out(out_text: str) -> float | None:
    """Return the raw printed MTC value when MASTER labels it recognisably.

    Only unit-tagged matches ("... PCM/C") are accepted — an untagged number
    could be the echoed input card, not a result (MOCHA sdm_mtc_parse L40-57).
    Callers convert with :func:`mtc_text_to_pcm_per_c` and an explicit unit.
    """

    hits: list[float] = []
    for pat in _MTC_PATTERNS:
        for match in pat.finditer(out_text):
            try:
                hits.append(float(match.group(1)))
            except ValueError:
                pass
    return hits[-1] if hits else None


def mtc_two_point_from_rows(keffs: Sequence[float], delta_c: float) -> float | None:
    """Central-difference MTC (pcm/°C) from the EXE_RHO branch keff rows.

    The MTC branch writes THREE reactivity rows ``[base, T+Δ, T−Δ]``; the
    MASTER-printed MTC matches ``(ρ(T+Δ) − ρ(T−Δ)) / (2·Δ)`` in sign and
    magnitude (MOCHA sdm_mtc_parse L77-91, verified 2026-07-02 smoke test — the
    naive ``(first,last)/Δ`` pairing flips the sign).  Anything but exactly 3
    rows is ambiguous: return ``None`` and let the caller warn.
    """

    if float(delta_c) == 0.0:
        raise ValueError("MTC two-point delta_c must be non-zero")
    if len(keffs) != 3:
        return None
    return (rho_pcm(keffs[1]) - rho_pcm(keffs[2])) / (2.0 * float(delta_c))


# --------------------------------------------------------------------------- #
# MASTER MAS_SUM branch reactivity parser (EDIT 2 K-EFF rows)
# --------------------------------------------------------------------------- #
# lpopt's vendor ``parse_mas_sum`` validates a *depletion* summary (EDIT 2/3 with
# a monotone EFPD grid + convergence gate) and does not expose per-row K-EFF, so
# a steady-state branch summary needs this small dedicated reader.  Format
# (MOCHA master_sum.py, verified against depf_*.sum):
#     SUMMARY EDIT 2 : REACTIVITY
#     NO. DAY EFPD CYC-BU TOT-BU P(%) PPM K-EFF ERRFLX REACT.
_EDIT_HEADER_RE = re.compile(r"^\s*SUMMARY\s+EDIT\s+(\d+)\s*:", re.I)


def _edit2_lines(sum_text: str) -> list[str]:
    """Lines belonging to ``SUMMARY EDIT 2`` (up to the next EDIT header)."""

    out: list[str] = []
    collecting = False
    for line in sum_text.splitlines():
        header = _EDIT_HEADER_RE.match(line)
        if header is not None:
            collecting = header.group(1) == "2"
            continue
        if collecting:
            out.append(line)
    return out


def parse_branch_keffs(sum_text: str) -> list[float]:
    """Per-branch multiplication factors from a MASTER branch ``MAS_SUM``.

    Each steady-state ``%EXE_STD`` / ``%EXE_RHO`` point emits one EDIT-2
    reactivity row; the K-EFF column is field index 7 (``NO DAY EFPD CYC-BU
    TOT-BU P(%) PPM K-EFF ...``).  Rows are returned in emission order.
    """

    keffs: list[float] = []
    for line in _edit2_lines(sum_text):
        toks = line.split()
        # NO DAY EFPD CYC-BU TOT-BU P(%) PPM K-EFF ERRFLX REACT.  -> >= 8 fields
        if len(toks) < 8 or not toks[0].isdigit():
            continue
        try:
            keffs.append(float(toks[7]))
        except ValueError:
            continue
    if not keffs:
        raise ValueError("branch MAS_SUM parse failure: no EDIT-2 K-EFF rows found")
    return keffs


# --------------------------------------------------------------------------- #
# rod model (port of MOCHA/sdm_mtc_io RodGroup/RodModel + APR1400 DCD counts)
# --------------------------------------------------------------------------- #
#: APR1400 DCD Figure 4.3-36 CEA bank populations (MOCHA sdm_mtc_io L24-34).
EXPECTED_DCD_ROD_COUNTS = {
    "A": 16, "B": 20, "R1": 8, "R2": 12, "R3": 12, "R4": 8, "R5": 5, "P": 12, "S": 8,
}
#: master4.0m4_r1 crashes when %ROD_CFG declares more than this many groups
#: (MOCHA sdm_mtc_io L267, located by bisection 92 OK / 93 crash).
MASTER_MAX_ROD_GROUPS = 92


@dataclass(frozen=True)
class RodGroup:
    """One control-rod group: id, bank, core box, finger count."""

    rod_id: str
    bank: str
    box: int = 0
    fingers: int = 12


@dataclass
class RodModel:
    """Full-core rod configuration for SDM branch synthesis.

    ``groups`` are the control-rod groups; ``rod_map`` is the JAGGED fuel-region
    map (rows follow the core boundary, reflector padding is 'o'), matching how
    MASTER reads ``%ROD_MAP`` (MOCHA sdm_mtc_io L250-259).  A licensing-grade
    SDM run requires the DCD-derived full-core model; this port takes it as an
    explicit input rather than re-parsing the DCD PDF geometry at deck time
    (composition-table-agnostic — see module docstring).
    """

    groups: list[RodGroup]
    rod_map: list[list[str]]

    def groups_for_banks(self, banks: Sequence[str]) -> list[RodGroup]:
        wanted = set(banks)
        return [g for g in self.groups if g.bank in wanted]


# --------------------------------------------------------------------------- #
# licensing limits + typed results (port of MOCHA/sdm_mtc_types.py)
# --------------------------------------------------------------------------- #
@dataclass
class LicensingLimits:
    """MTC window + SDM required allowance (APR1400 DCD Table 4.3 defaults).

    Mirrors MOCHA sdm_mtc_types.LicensingLimits L9-25 / sdm_mtc_io
    ``extract_dcd_limits`` L84-128.  ``mtc_max_pcm_per_c`` is the most-positive
    allowed MTC; ``sdm_required_pcm`` is the minimum required net worth.
    """

    mtc_min_pcm_per_c: float = -54.0
    mtc_max_pcm_per_c: float = 9.0
    sdm_required_pcm: float = 10870.0
    cea_allowance_pcm: float = 10180.0
    net_worth_uncertainty_pcm: float = 690.0
    dcd_all_cea_worth_pcm: float = 16700.0
    dcd_stuck_worth_pcm: float = 5690.0
    dcd_excess_pcm: float = 140.0
    limits_source: str = "dcd_constant"
    #: Whether a measured value may be turned into a PASS/FAIL verdict at all
    #: (decision D9 / ``[constraints]``).  ``False`` = REPORT-ONLY: the branch runs,
    #: the number is recorded, and ``pass_limit`` stays ``None`` — nothing is ever
    #: marked a violator against a limit the user has not set.  Defaults ``True``
    #: so a hand-built ``LicensingLimits`` keeps the DCD-constant judging behaviour;
    #: :meth:`from_constraints` is what makes a deck default to report-only.
    mtc_gated: bool = True
    sdm_gated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_constraints(
        cls, constraints: Any, sdm_mtc_cfg: Any = None
    ) -> "LicensingLimits":
        """Build limits from a ``[constraints]`` section (report-only by default).

        Every user-set limit overrides; every UNSET limit falls back to the
        APR1400 DCD constant **for reporting context only** and leaves that axis
        UNGATED, so the report can say "measured -23.1 pcm/°C; the DCD-standard
        window is [-54, +9] but no user limit is set" without ever converting a
        suggestion into a verdict.  ``sdm_mtc_cfg`` (the ``[sdm_mtc]`` mechanics
        section) supplies the CEA-allowance breakdown when present.
        """

        limits = cls()
        if sdm_mtc_cfg is not None:
            for attr in ("cea_allowance_pcm", "net_worth_uncertainty_pcm"):
                value = getattr(sdm_mtc_cfg, attr, None)
                if value is not None:
                    setattr(limits, attr, float(value))
        if constraints is None:
            return limits
        mtc_max = getattr(constraints, "mtc_max_pcm_per_c", None)
        mtc_min = getattr(constraints, "mtc_min_pcm_per_c", None)
        sdm_req = getattr(constraints, "sdm_required_pcm", None)
        if mtc_max is not None:
            limits.mtc_max_pcm_per_c = float(mtc_max)
        if mtc_min is not None:
            limits.mtc_min_pcm_per_c = float(mtc_min)
        if sdm_req is not None:
            limits.sdm_required_pcm = float(sdm_req)
        gated_mtc = getattr(constraints, "mtc_gated", None)
        gated_sdm = getattr(constraints, "sdm_gated", None)
        limits.mtc_gated = bool(gated_mtc()) if callable(gated_mtc) else bool(gated_mtc)
        limits.sdm_gated = bool(gated_sdm()) if callable(gated_sdm) else bool(gated_sdm)
        set_axes = [name for name, gated in
                    (("MTC", limits.mtc_gated), ("SDM", limits.sdm_gated)) if gated]
        limits.limits_source = (
            "user[constraints]:" + "+".join(set_axes) if set_axes
            else "report_only (no user limit set; DCD constants shown for context)"
        )
        return limits

    @classmethod
    def from_overrides(
        cls,
        base: "LicensingLimits | None" = None,
        *,
        mtc_limit: float | None = None,
        sdm_limit: float | None = None,
    ) -> "LicensingLimits":
        """Apply CLI/config overrides (``mtc_limit`` = most-positive MTC;
        ``sdm_limit`` = minimum required SDM) onto a base limits set."""

        limits = cls(**base.to_dict()) if base is not None else cls()
        if mtc_limit is not None:
            limits.mtc_max_pcm_per_c = float(mtc_limit)
            limits.limits_source = "override"
        if sdm_limit is not None:
            limits.sdm_required_pcm = float(sdm_limit)
            limits.limits_source = "override"
        return limits


@dataclass
class MtcResult:
    state: str
    value_pcm_per_c: float | None = None
    pass_limit: bool | None = None
    source: str = ""
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RodWorth:
    rod_id: str
    bank: str
    worth_pcm: float
    rho_stuck_pcm: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SdmResult:
    state: str
    rho_aro_pcm: float
    rho_ari_pcm: float
    w_ari_pcm: float
    worst_stuck: RodWorth | None
    required_pcm: float
    available_pcm: float
    margin_pcm: float
    pass_limit: bool | None
    rod_worths: list[RodWorth] = field(default_factory=list)
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["worst_stuck"] = self.worst_stuck.to_dict() if self.worst_stuck else None
        d["rod_worths"] = [r.to_dict() for r in self.rod_worths]
        return d


@dataclass
class SdmMtcResult:
    """One candidate's licensing verdict (the run_post_verification unit)."""

    record_id: str
    tag: str
    mtc: MtcResult | None = None
    sdm: SdmResult | None = None
    pass_mtc: bool | None = None
    pass_sdm: bool | None = None
    decks: dict[str, str] = field(default_factory=dict)   # name -> deck path/text ref
    wall: float = 0.0
    status: str = "pending"
    failure: str = ""
    #: MASTER branch processes actually launched for THIS candidate.  Counted, not
    #: assumed: a failed MTC branch still spent its call, and an SDM branch that was
    #: never built spent none.  The campaign sums these into the licensing budget.
    master_calls: int = 0
    #: True when every enabled axis ran report-only (no user limit set), so the
    #: absence of a FAIL is a measurement, not a clearance.
    report_only: bool = False

    @property
    def passed(self) -> bool:
        """Overall PASS: BOTH constraints an explicit pass.

        An indeterminate (``None``) MTC/SDM row is never a pass — evidence
        absence is never a pass (MOCHA sdm_mtc_types ``pass_limit`` L84-95,
        audit T7-1).
        """

        return self.pass_mtc is True and self.pass_sdm is True

    @property
    def verdict(self) -> str:
        """Honest four-way verdict for the table.

        ``ERR`` (run failed) / ``FAIL`` (an explicit constraint violation) /
        ``PASS`` (both explicit passes) / ``REPORT`` (measured successfully but no
        user limit to judge against — decision D9 report-only default) /
        ``INCONCLUSIVE`` (no violation but an indeterminate constraint, e.g. SDM
        deferred for lack of a rod model).
        """

        if self.status != "ok":
            return "ERR"
        if self.pass_mtc is False or self.pass_sdm is False:
            return "FAIL"
        if self.pass_mtc is True and self.pass_sdm is True:
            return "PASS"
        if self.report_only:
            return "REPORT"
        return "INCONCLUSIVE"

    @property
    def violates(self) -> bool:
        """True iff a GATED axis was explicitly violated.

        This — not ``not passed`` — is the delivery-marking predicate: an
        unmeasured or report-only candidate is not a violator, and must not be
        dropped as though it were.
        """

        return self.pass_mtc is False or self.pass_sdm is False

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "tag": self.tag,
            "mtc": self.mtc.to_dict() if self.mtc else None,
            "sdm": self.sdm.to_dict() if self.sdm else None,
            "pass_mtc": self.pass_mtc,
            "pass_sdm": self.pass_sdm,
            "pass": self.passed,
            "violates": self.violates,
            "verdict": self.verdict,
            "report_only": self.report_only,
            "master_calls": self.master_calls,
            "decks": self.decks,
            "wall": self.wall,
            "status": self.status,
            "failure": self.failure,
        }


# --------------------------------------------------------------------------- #
# candidate reference + branch parameters
# --------------------------------------------------------------------------- #
@dataclass
class CandidateRef:
    """A converged, spec-feasible candidate to licence-verify.

    ``deck_text`` is the candidate's final-cycle MASTER deck (the WaveVerifier
    staged deck); ``restart_path`` is the converged equilibrium restart the
    branches read (explicit — the caller resolves the WaveVerifier's
    kept/promoted final restart).  ``metrics`` carries the campaign FOM for the
    verdict table.
    """

    record_id: str
    tag: str
    deck_text: str
    restart_path: Path
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_paths(
        cls,
        record_id: str,
        tag: str,
        deck_path: str | Path,
        restart_path: str | Path,
        metrics: Mapping[str, Any] | None = None,
    ) -> "CandidateRef":
        return cls(
            record_id=record_id,
            tag=tag,
            deck_text=_read_deck_flex(Path(deck_path)),
            restart_path=Path(restart_path),
            metrics=dict(metrics or {}),
        )


@dataclass
class BranchParams:
    """Branch-synthesis parameters (states, temperature step, rod banks).

    * ``states`` — condition suffixes; ``*hfp`` → power load 1.0, ``*hzp`` →
      1e-6 (MOCHA sdm_mtc_io ``_state_pload`` L314-315).  MTC defaults to HFP
      (Δρ/ΔT_mod around the operating temperature); SDM defaults to HZP
      (shutdown margin at hot-zero-power).
    * ``mtc_delta_c`` — moderator-temperature step ΔT_mod for the ``%EXE_RHO``
      card (kept coupled to the two-point scale — audit T7-2).
    * ``mtc_output_units`` — unit of the MASTER-printed MTC text
      (``pcm_per_c`` | ``drho_per_c_1e-4``).
    * ``scram_banks`` / ``stuck_candidate_banks`` — SDM rod banks; ``rod_model``
      is the full-core %ROD_CFG/%ROD_MAP source (required for SDM).
    """

    states: tuple[str, ...] = ()
    mtc_delta_c: float = 5.0
    mtc_output_units: str = "pcm_per_c"
    # A and B are the SHUTDOWN banks and carry ~74 % of total CEA worth; omitting
    # them makes the 10,870 pcm DCD requirement unreachable for every pattern.
    # Kept in sync with ``SdmMtcConfig`` (lpopt/config.py) and the validated
    # MOCHA scope.  PSCEA ``P`` remains excluded by default.
    scram_banks: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "B", "A")
    stuck_candidate_banks: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "B", "A")
    rod_model: RodModel | None = None

    def states_for(self, kind: str) -> tuple[str, ...]:
        if self.states:
            return self.states
        return ("hfp",) if kind == "mtc" else ("hzp",)


def _state_pload(state: str) -> float:
    """HFP → full power (1.0); HZP → ~zero power (1e-6).  MOCHA L314-315."""

    return 1.0 if state.endswith("hfp") else 1.0e-6


# --------------------------------------------------------------------------- #
# %ROD_CFG / %ROD_MAP / %EXE_ROD builders (port of MOCHA/sdm_mtc_io.py)
# --------------------------------------------------------------------------- #
def _rod_cfg_block(groups: Sequence[RodGroup]) -> str:
    """%ROD_CFG for ``groups`` (MOCHA sdm_mtc_io rod_cfg_block L292-299)."""

    if len(groups) > MASTER_MAX_ROD_GROUPS:
        raise ValueError(
            f"SDM deck needs {len(groups)} rod groups, above the master4.0m4_r1 "
            f"limit of {MASTER_MAX_ROD_GROUPS} — drop a bank from scram/stuck banks"
        )
    lines = ["%ROD_CFG", f"        {len(groups)}"]
    lines.append(
        "# idgr, mattip, matabs, matfol, lentip, lenabs, crups, crlos, "
        "crpos1, crpos2, finger"
    )
    for g in groups:
        lines.append(
            f"        {g.rod_id:<4} 0     1     0     0     381.  411.  30.   "
            f"381.  381.  {g.fingers}"
        )
    return "\n".join(lines) + "\n"


def _rod_map_block(rod_map: Sequence[Sequence[str]], groups: Sequence[RodGroup]) -> str:
    """%ROD_MAP restricted to ``groups`` (MOCHA sdm_mtc_io rod_map_block L302-311)."""

    ids = {g.rod_id for g in groups}
    lines = ["%ROD_MAP"]
    for row in rod_map:
        toks = [(t if t in ids else "o") for t in row]
        lines.append("        " + "  ".join(f"{tok:<4}" for tok in toks).rstrip())
    return "\n".join(lines) + "\n"


def _exe_rod_block(lines: Sequence[str]) -> str:
    """%EXE_ROD card (MOCHA sdm_mtc_io _exe_rod_block L332-335)."""

    if not lines:
        return "%EXE_ROD\n\n"
    return "%EXE_ROD\n" + "\n".join(f"        {ln}" for ln in lines) + "\n"


def _insert_lines(
    scram_banks: Sequence[str],
    stuck_id: str | None = None,
    known_banks: set[str] | None = None,
) -> list[str]:
    """EXE_ROD lines inserting the scram banks (one line per bank), then holding
    one rod out (MOCHA sdm_mtc_io _insert_lines L338-358).

    MASTER positions every group whose id BEGINS with the given token (crtype
    prefix match), so one line per bank ("R1  0") inserts exactly that bank —
    the collapsed "R  0" would insert ALL R banks and overstate scram worth
    (audit T7-4).  The stuck-rod line comes last: later EXE_ROD lines redefine
    earlier prefix matches.
    """

    out: list[str] = []
    for bank in dict.fromkeys(str(b) for b in scram_banks):
        if known_banks is not None and bank not in known_banks:
            raise ValueError(
                f"scram/stuck bank {bank!r} matches no rod bank {sorted(known_banks)}"
            )
        out.append(f"{bank}  0")
    if stuck_id:
        out.append(f"{stuck_id}  381.")
    return out


# --------------------------------------------------------------------------- #
# deck synthesis — text passthrough, swap only restart ref + EXE/branch section
# --------------------------------------------------------------------------- #
_EXE_CARD_RE = re.compile(r"^[ \t]*%(?:EXE_[A-Z0-9]+|EDT_OPT)\b", re.IGNORECASE | re.MULTILINE)
_ROD_CARD_RE = re.compile(r"^[ \t]*%ROD_(?:CFG|MAP)\b", re.IGNORECASE | re.MULTILINE)


def _read_deck_flex(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decoding is total")


def _execution_span(deck: str) -> int:
    """Byte offset of the deck's execution section start.

    The execution section is everything from the first ``%EXE_*`` / ``%EDT_OPT``
    card (or an earlier ``%ROD_CFG`` / ``%ROD_MAP`` if the base deck already
    carries rod cards) through end-of-deck.  Everything before it — the whole
    geometry / composition / loading-pattern definition (``%JOB_*`` …
    ``%LPD_HFF``) — is preserved byte-for-byte, the same philosophy as
    :func:`replace_lpd_shf` swapping only the ``%LPD_SHF`` body.
    """

    exe = _EXE_CARD_RE.search(deck)
    if exe is None:
        raise ValueError("deck has no %EXE_* / %EDT_OPT execution section to swap")
    start = exe.start()
    rod = _ROD_CARD_RE.search(deck)
    if rod is not None and rod.start() < start:
        start = rod.start()
    return start


def _swap_execution(deck: str, restart_basename: str, branch_body: str) -> str:
    """Return ``deck`` with the restart reference and execution section swapped.

    Two byte-scoped edits: (1) :func:`advance_cycle_deck` rewrites *only* the
    sole ``%JOB_TYP`` restart reference to ``restart_basename`` (keeping the
    deck's own cycle, so ``%JOB_IDE`` / a matching ``CYnn`` title stay put);
    (2) everything from the first execution card to EOF is replaced by
    ``branch_body``.  The preserved prefix is byte-identical to the base deck.
    """

    deck = advance_cycle_deck(deck, restart_basename, deck_cycle(deck))
    start = _execution_span(deck)
    prefix = deck[:start]
    if not prefix.endswith("\n"):
        prefix += "\n"
    return prefix + branch_body


def _mtc_body(state: str, params: BranchParams) -> str:
    """The MTC branch execution cards (MOCHA build_mtc_branch_full L400-419).

    ARO (rodless): the deck carries no rod cards, so the branch is physically an
    all-rods-out steady state.  ``%EXE_STD boron eq`` fixes the operating point,
    ``%EXE_RHO mtc 1`` perturbs T_mod by ±ΔT_mod and prints/records the two-point
    reactivity difference (``dtm`` column, MOCHA _exe_rho_mtc L421-432).
    """

    pload = _state_pload(state)
    dtm = float(params.mtc_delta_c)
    return (
        "%EXE_STD\n"
        f"        boron   eq      tr      {pload:g}\n"
        "%EXE_RHO\n"
        "        mtc     1\n"
        f"        0   0   0   {dtm:g}   0   0   0"
        "                                        # dppm, dpow, dtf, dtm, ddm, dxe, dsm\n"
        "/\n"
        "END\n"
    )


def sdm_branch_labels(params: BranchParams) -> list[str]:
    """Deterministic branch-emission labels for one SDM deck.

    Order: ``aro_critical``, ``ari`` (all-rods-in), then one ``stuck:<id>:<bank>``
    per stuck-candidate rod.  Used by both synthesis and parsing so the keff-row
    order is unambiguous (MOCHA build_sdm_branch L369-395).
    """

    if params.rod_model is None:
        raise ValueError("SDM branch requires params.rod_model (full-core %ROD_MAP)")
    labels = ["aro_critical", "ari"]
    for rod in params.rod_model.groups_for_banks(params.stuck_candidate_banks):
        labels.append(f"stuck:{rod.rod_id}:{rod.bank}")
    return labels


def _sdm_body(state: str, params: BranchParams) -> str:
    """The SDM branch execution cards (MOCHA build_sdm_branch L361-397).

    One deck solves ``2 + n_stuck`` steady states in a single MASTER run:
    ARO-critical → ARI (scram banks inserted) → one case per stuck rod (scram
    banks inserted, that rod held fully withdrawn).  The keff of each gives the
    reactivity, and SDM = W_ari − worst_stuck − required.
    """

    model = params.rod_model
    if model is None:
        raise ValueError("SDM branch requires params.rod_model (full-core %ROD_MAP)")
    banks = {g.bank for g in model.groups}
    active = model.groups_for_banks(set(params.scram_banks) | set(params.stuck_candidate_banks))
    if not active:
        raise ValueError("scram/stuck banks match no rod group in rod_model")
    pload = _state_pload(state)

    body = [
        _rod_cfg_block(active),
        _rod_map_block(model.rod_map, active),
        _exe_rod_block([]),                       # ARO (all rods out)
        f"%EXE_STD\n        boron   tr      tr      {pload:g}\n/\n",
        _exe_rod_block(_insert_lines(params.scram_banks, known_banks=banks)),  # ARI
        f"%EXE_STD\n        keff    tr      tr      {pload:g}\n/\n",
    ]
    for rod in model.groups_for_banks(params.stuck_candidate_banks):
        body.append(
            _exe_rod_block(
                _insert_lines(params.scram_banks, stuck_id=rod.rod_id, known_banks=banks)
            )
        )
        body.append(f"%EXE_STD\n        keff    tr      tr      {pload:g}\n/\n")
    body.append("END\n")
    return "".join(body)


def build_branch_decks(
    base_deck_text: str,
    restart_basename: str,
    kind: str,
    params: BranchParams,
) -> list[tuple[str, str]]:
    """Synthesise the SDM/MTC branch decks for one candidate.

    ``base_deck_text`` is the candidate's converged final-cycle deck;
    ``restart_basename`` is the converged restart the branches read.  For each
    configured state one ``(name, deck_text)`` is returned.  The synthesis is
    **text passthrough**: the entire geometry / composition / loading-pattern
    prefix of the base deck is preserved byte-for-byte; only the ``%JOB_TYP``
    restart reference and the execution (``%EXE_*`` / ``%ROD_*``) section are
    swapped — the same philosophy as ``replace_lpd_shf`` (composition-table
    agnostic).  ``kind`` is ``"mtc"`` or ``"sdm"``.
    """

    kind = str(kind).lower()
    if kind not in ("mtc", "sdm"):
        raise ValueError("kind must be 'mtc' or 'sdm'")
    decks: list[tuple[str, str]] = []
    for state in params.states_for(kind):
        body = _mtc_body(state, params) if kind == "mtc" else _sdm_body(state, params)
        deck_text = _swap_execution(base_deck_text, restart_basename, body)
        decks.append((f"{kind}_{state}", deck_text))
    return decks


# --------------------------------------------------------------------------- #
# branch execution — reuse vendor MasterRunner staging mechanics
# --------------------------------------------------------------------------- #
@dataclass
class BranchOutputs:
    """Raw text of one branch MASTER run's MAS_SUM + MAS_OUT."""

    sum_text: str
    out_text: str
    wall: float = 0.0
    work_dir: Path | None = None


class BranchExecutor:
    """Protocol for running one branch deck.  Tests inject a fake (no MASTER)."""

    def run_branch(self, name: str, deck_text: str, restart_path: Path) -> BranchOutputs:
        raise NotImplementedError


class VendorBranchExecutor(BranchExecutor, MasterRunner):
    """Branch runner reusing :class:`MasterRunner` staging mechanics.

    Inherits the vendor asset resolution / restart validation (``_assets``),
    file staging (``_link_or_copy``), CPU-affinity pinning
    (``_apply_cpu_affinity``) and timeout handling — but keeps ``MAS_SUM`` +
    ``MAS_OUT`` text instead of running the depletion-summary parser + metric
    cache (a steady-state branch summary has no EFPD grid, so the vendor
    ``parse_mas_sum`` would reject it).  No equilibrium chain is run.
    """

    def run_branch(self, name: str, deck_text: str, restart_path: Path) -> BranchOutputs:
        # Reuse the vendor asset resolver: validates xsl/hff presence, the
        # restart override, and that the deck references the restart by name.
        deck, restart, xsl, hff = self._assets(self.work_root, deck_text, restart_path)

        self.work_root.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:40] or "branch"
        work_dir = Path(
            tempfile.mkdtemp(prefix=f"{safe}-", dir=self.work_root)
        ).resolve()
        t0 = time.perf_counter()
        try:
            (work_dir / "MAS_INP").write_bytes(deck.encode("ascii"))
            self._link_or_copy(xsl, work_dir / "MAS_XSL")
            self._link_or_copy(hff, work_dir / "MAS_HFF")
            self._link_or_copy(restart, work_dir / restart.name)

            stdout_path = work_dir / "MASTER.stdout"
            stderr_path = work_dir / "MASTER.stderr"
            with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
                self.process_calls += 1
                proc = subprocess.Popen(self.command, cwd=work_dir, stdout=so, stderr=se,
                                        **no_window_flags())
                try:
                    self._apply_cpu_affinity(proc)
                except Exception:
                    proc.kill()
                    proc.wait()
                    raise
                try:
                    rc = proc.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired as exc:
                    proc.kill()
                    proc.wait()
                    raise MasterRunError(
                        f"branch MASTER timed out after {self.timeout:g}s",
                        work_dir=work_dir,
                    ) from exc
            if rc != 0:
                raise MasterRunError(f"branch MASTER exited with status {rc}", work_dir=work_dir)
            sum_file = work_dir / "MAS_SUM"
            if not sum_file.is_file() or sum_file.stat().st_size == 0:
                raise MasterRunError("branch MASTER produced no MAS_SUM", work_dir=work_dir)
            out_file = work_dir / "MAS_OUT"
            outputs = BranchOutputs(
                sum_text=_read_deck_flex(sum_file),
                out_text=_read_deck_flex(out_file) if out_file.is_file() else "",
                wall=time.perf_counter() - t0,
                work_dir=work_dir if self.keep_success else None,
            )
        except MasterRunError:
            raise  # work dir already retained for diagnosis
        except Exception as exc:
            raise MasterRunError(
                f"unexpected branch integration failure: {exc}", work_dir=work_dir
            ) from exc
        if not self.keep_success:
            shutil.rmtree(work_dir, ignore_errors=True)
        return outputs


# --------------------------------------------------------------------------- #
# per-candidate evaluation
# --------------------------------------------------------------------------- #
def _evaluate_mtc(
    outputs: BranchOutputs, state: str, limits: LicensingLimits, params: BranchParams
) -> MtcResult:
    """Compute MTC (pcm/°C) + pass verdict from a branch's outputs.

    Primary: unit-safe keff two-point central difference (audit F1/F8).
    Fallback: the MASTER-printed ``... PCM/C`` text via the audit-corrected
    :func:`mtc_text_to_pcm_per_c` (no magnitude heuristic).
    """

    warning = ""
    value: float | None = None
    source = ""
    try:
        keffs = parse_branch_keffs(outputs.sum_text)
    except ValueError as exc:
        keffs = []
        warning = str(exc)
    value = mtc_two_point_from_rows(keffs, params.mtc_delta_c) if len(keffs) == 3 else None
    if value is not None:
        source = "two_point_keff"
    else:
        parsed = parse_mtc_from_out(outputs.out_text)
        if parsed is not None:
            value = mtc_text_to_pcm_per_c(parsed, params.mtc_output_units)
            source = f"EXE_RHO_text[{params.mtc_output_units}]"
            if len(keffs) != 3:
                warning = (
                    f"two-point needs the 3-row [base,T+Δ,T-Δ] keff pattern "
                    f"(got {len(keffs)}); used the MASTER-printed MTC text"
                )
        else:
            warning = warning or (
                f"no MTC evidence: two-point needs 3 keff rows (got {len(keffs)}) "
                f"and no unit-tagged 'PCM/C' line was found in MAS_OUT"
            )
    passed: bool | None = None
    if value is not None and limits.mtc_gated:
        passed = limits.mtc_min_pcm_per_c <= value <= limits.mtc_max_pcm_per_c
    elif value is not None:
        # Measured but not judged — the D9 default until the user sets a limit.
        # ``pass_limit`` stays None so nothing downstream can read the silence as
        # a clearance (SdmMtcResult.passed requires an explicit True).
        warning = "; ".join(filter(None, [
            warning,
            f"REPORT-ONLY: no user MTC limit set ([constraints] mtc_max_pcm_per_c); "
            f"measured {value:.2f} pcm/°C, DCD-standard window is "
            f"[{limits.mtc_min_pcm_per_c:g}, {limits.mtc_max_pcm_per_c:g}]",
        ]))
    return MtcResult(state=state, value_pcm_per_c=value, pass_limit=passed,
                     source=source, warning=warning)


def _evaluate_sdm(
    outputs: BranchOutputs, state: str, limits: LicensingLimits, params: BranchParams
) -> SdmResult:
    """Compute SDM (pcm) + pass verdict from a branch's keff rows (MOCHA _run_sdm)."""

    labels = sdm_branch_labels(params)
    keffs = parse_branch_keffs(outputs.sum_text)
    if len(keffs) < len(labels):
        raise ValueError(
            f"SDM branch parse mismatch: {len(keffs)} keff rows for {len(labels)} labels"
        )
    rhos = [rho_pcm(k) for k in keffs[: len(labels)]]
    rho_aro, rho_ari = rhos[0], rhos[1]
    w_ari = rho_aro - rho_ari
    worths: list[RodWorth] = []
    for label, rho_stuck in zip(labels[2:], rhos[2:]):
        _prefix, rod_id, bank = label.split(":")
        worths.append(
            RodWorth(rod_id=rod_id, bank=bank, worth_pcm=rho_stuck - rho_ari,
                     rho_stuck_pcm=rho_stuck)
        )
    if not worths:
        raise ValueError("SDM branch produced no stuck-rod cases")
    worst = max(worths, key=lambda r: r.worth_pcm)
    available = w_ari - worst.worth_pcm
    margin = available - limits.sdm_required_pcm
    # Every stuck-rod branch must sit between ARI (all in) and ARO (all out);
    # checking only the max-rho rod hides inversions on the others (audit T7-9).
    bad = [w.rod_id for w in worths if not (rho_aro >= w.rho_stuck_pcm >= rho_ari)]
    warning = (
        "ARO >= stuck >= ARI monotonicity failed for rod(s): " + ", ".join(bad)
        if bad else ""
    )
    passed: bool | None = (margin >= 0.0) if limits.sdm_gated else None
    if not limits.sdm_gated:
        warning = "; ".join(filter(None, [
            warning,
            f"REPORT-ONLY: no user SDM limit set ([constraints] sdm_required_pcm); "
            f"measured available worth {available:.0f} pcm, DCD-standard "
            f"requirement is {limits.sdm_required_pcm:g} pcm",
        ]))
    return SdmResult(
        state=state, rho_aro_pcm=rho_aro, rho_ari_pcm=rho_ari, w_ari_pcm=w_ari,
        worst_stuck=worst, required_pcm=limits.sdm_required_pcm,
        available_pcm=available, margin_pcm=margin, pass_limit=passed,
        rod_worths=worths, warning=warning,
    )


def _verify_one(
    candidate: CandidateRef,
    limits: LicensingLimits,
    mtc_params: BranchParams | None,
    sdm_params: BranchParams | None,
    executor: BranchExecutor,
    keep_decks: bool,
    deck_dir: Path | None,
) -> SdmMtcResult:
    result = SdmMtcResult(record_id=candidate.record_id, tag=candidate.tag)
    t0 = time.perf_counter()
    restart_name = Path(candidate.restart_path).name
    ran_mtc = ran_sdm = False

    def _run(name: str, deck_text: str) -> BranchOutputs:
        """Stage + run one branch, charging the MASTER call BEFORE it can fail.

        A branch that times out or crashes still consumed a MASTER process; if the
        counter were incremented after a successful return, the licensing budget
        would silently under-report exactly the runs that cost the most wall time.
        """

        if deck_dir is not None:
            path = deck_dir / f"{candidate.tag}__{name}.inp"
            path.write_text(deck_text, encoding="ascii")
            result.decks[name] = str(path)
        result.master_calls += 1
        return executor.run_branch(name, deck_text, candidate.restart_path)

    try:
        # --- MTC branch(es) -------------------------------------------------- #
        if mtc_params is not None:
            mtc_decks = build_branch_decks(
                candidate.deck_text, restart_name, "mtc", mtc_params
            )
            for name, deck_text in mtc_decks:
                outputs = _run(name, deck_text)
                state = name.split("_", 1)[1]
                result.mtc = _evaluate_mtc(outputs, state, limits, mtc_params)
            ran_mtc = result.mtc is not None
        result.pass_mtc = result.mtc.pass_limit if result.mtc else None

        # --- SDM branch(es) -------------------------------------------------- #
        if sdm_params is not None and sdm_params.rod_model is not None:
            sdm_decks = build_branch_decks(candidate.deck_text, restart_name, "sdm", sdm_params)
            for name, deck_text in sdm_decks:
                outputs = _run(name, deck_text)
                state = name.split("_", 1)[1]
                result.sdm = _evaluate_sdm(outputs, state, limits, sdm_params)
            result.pass_sdm = result.sdm.pass_limit if result.sdm else None
            ran_sdm = result.sdm is not None
        else:
            result.pass_sdm = None
            if result.sdm is None:
                # No rod model -> SDM cannot be evaluated; never a silent pass.
                result.failure = (
                    "SDM skipped: no rod_model supplied (full-core %ROD_CFG/%ROD_MAP "
                    "required for a licensing-grade shutdown-margin branch)"
                )
        # Report-only iff at least one axis produced a number and NO axis produced
        # a verdict: that is "measured, not judged", distinct from "not measured".
        result.report_only = bool(
            (ran_mtc or ran_sdm)
            and result.pass_mtc is None and result.pass_sdm is None
        )
        result.status = "ok"
    except Exception as exc:  # noqa: BLE001 — isolate one candidate's failure
        result.status = "failed"
        result.failure = f"{type(exc).__name__}: {exc}"
    result.wall = time.perf_counter() - t0
    if not keep_decks:
        result.decks = {}
    return result


def run_post_verification(
    candidates: Sequence[CandidateRef],
    limits: LicensingLimits,
    master_cfg: Mapping[str, Any] | None = None,
    work_root: str | Path = "runs/sdm_mtc_work",
    *,
    mtc_params: BranchParams | None = None,
    sdm_params: BranchParams | None = None,
    executor: BranchExecutor | None = None,
    sidecar_path: str | Path | None = DEFAULT_SIDECAR,
    keep_decks: bool = True,
    run_mtc: bool = True,
) -> list[SdmMtcResult]:
    """Licence-verify a list of candidates (≈1 MASTER branch call per enabled axis).

    ``master_cfg`` supplies the default (real-MASTER) executor keys
    (``executable``, ``package_root``, ``timeout``, ``keep_success``,
    ``cpu_core``); tests inject ``executor`` (a fake) instead and skip MASTER
    entirely.  ``run_mtc=False`` disables the MTC axis outright (``[constraints]``
    ``mtc_enable = false``) — and disabled means *no branch is launched*, so the
    call count reflects it.  Every result is appended to the
    ``data/sdm_mtc/results.jsonl`` sidecar (keyed by ``record_id``, deduped)
    unless ``sidecar_path`` is None.
    """

    work_root = Path(work_root)
    mtc_params = (mtc_params or BranchParams()) if run_mtc else None
    limits = limits or LicensingLimits()

    if executor is None:
        cfg = dict(master_cfg or {})
        if not cfg.get("executable") or not cfg.get("package_root"):
            raise ValueError(
                "run_post_verification needs master_cfg {executable, package_root} "
                "for the real-MASTER executor, or an injected executor (tests)"
            )
        executor = VendorBranchExecutor(
            cfg["package_root"],
            cfg["executable"],
            work_root=work_root,
            cache_dir=work_root / "cache",
            timeout=float(cfg.get("timeout", 300.0)),
            keep_success=bool(cfg.get("keep_success", False)),
            cpu_core=cfg.get("cpu_core"),
        )

    deck_dir = (work_root / "decks") if keep_decks else None
    if deck_dir is not None:
        deck_dir.mkdir(parents=True, exist_ok=True)

    results: list[SdmMtcResult] = []
    for candidate in candidates:
        result = _verify_one(
            candidate, limits, mtc_params, sdm_params, executor, keep_decks, deck_dir
        )
        results.append(result)
        if sidecar_path is not None:
            append_sidecar(sidecar_path, result, limits=limits)
    return results


# --------------------------------------------------------------------------- #
# sidecar store (data/sdm_mtc/results.jsonl) — append + dedup by record_id
# --------------------------------------------------------------------------- #
def load_sidecar(path: str | Path) -> list[dict[str, Any]]:
    """Read the sidecar JSONL, deduped by ``record_id`` (last write wins)."""

    path = Path(path)
    if not path.is_file():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("record_id"))
            if rid not in by_id:
                order.append(rid)
            by_id[rid] = row
    return [by_id[rid] for rid in order]


def append_sidecar(
    path: str | Path,
    result: SdmMtcResult,
    *,
    limits: LicensingLimits | None = None,
) -> Path:
    """Append one result to the sidecar, replacing any prior row for the same
    ``record_id`` (dedup, last-write-wins).  The file is rewritten atomically so
    it never holds two rows for one candidate."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = result.to_dict()
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if limits is not None:
        row["limits"] = limits.to_dict()

    existing = load_sidecar(path)
    merged: dict[str, dict[str, Any]] = {str(r.get("record_id")): r for r in existing}
    merged[result.record_id] = row

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for r in merged.values():
            handle.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# campaign integration hook (additive; NOT wired into the live loop)
# --------------------------------------------------------------------------- #
def select_topk_feasible(run_dir: str | Path, top_k: int) -> list[CandidateRef]:
    """Pick the top-K feasible converged candidates archived under a run dir.

    Reads ``runs/<ts>/candidates/<pair>/<digest>/meta.json`` entries (as written
    by the WaveVerifier archive), keeps those with ``feasible == True`` and a
    resolvable restart, and orders them by proximity to the campaign objective:
    ``|cyclen - 625|`` when the candidate recorded a cycle length, else CBC.

    **The order is cyclen proximity, NOT F_xy** — that is a registered property,
    not an accident (slice-Z prereg "registered selection-order trap", tasks
    #21b).  On a ``min_fxy`` arm the PRIMARY candidate (minimum measured F_xy)
    can therefore fall outside the top-K; the registered response is to RAISE
    ``--top-k`` until it is inside and re-run (one extra MASTER call per extra
    candidate), NOT to re-rank here.  Re-ranking would silently change which
    cores get licence-verified and would invalidate the registered MASTER-call
    budget, so it must be a prereg amendment, never a code change.

    LEGACY selector, kept for ``lpopt sdm-mtc`` on runs that predate the
    provenance index.  The D9 gate uses :func:`candidates_from_delivery` instead,
    because the delivery ranking — not the archive — is what defines "top-K flat
    feasible".  Both refuse a candidate whose OWN converged restart is missing.
    """

    run_dir = Path(run_dir)
    metas: list[tuple[float, dict[str, Any], Path]] = []
    for meta_path in sorted((run_dir / "candidates").glob("*/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not meta.get("feasible"):
            continue
        cyclen = meta.get("cyclen")
        # smaller key = better (nearest 625 EFPD, else fall back to CBC).
        key = abs(float(cyclen) - 625.0) if cyclen is not None else float(meta.get("CBC_max", 1e9))
        metas.append((key, meta, meta_path.parent))
    metas.sort(key=lambda t: t[0])

    candidates: list[CandidateRef] = []
    for _key, meta, folder in metas[: max(int(top_k), 0)]:
        deck_path = _find_candidate_deck(folder, run_dir)
        restart_path = _find_candidate_restart(folder, run_dir)
        if deck_path is None or restart_path is None:
            continue
        rid = str((meta.get("extras") or {}).get("record_id") or folder.name)
        candidates.append(
            CandidateRef.from_paths(
                record_id=rid,
                tag=f"{folder.parent.name}_{folder.name[:8]}",
                deck_path=deck_path,
                restart_path=restart_path,
                metrics={k: meta.get(k) for k in
                         ("F_r", "F_q", "CBC_max", "cyclen", "AO_min", "AO_max")},
            )
        )
    return candidates


def _find_candidate_deck(folder: Path, run_dir: Path) -> Path | None:
    """The candidate's final-cycle MASTER deck (explicit or staged fallback)."""

    for cand in sorted(folder.glob("MAS_INP*.inp")):
        if cand.is_file():
            return cand
    # Fallback: the run's staged produce-case deck for this pair.
    pair = folder.parent.name
    staged = sorted((run_dir / "master" / "produce_cases" / pair).glob("*/MAS_INP*.inp"))
    return staged[0] if staged else None


def _find_candidate_restart(folder: Path, run_dir: Path) -> Path | None:
    """The candidate's OWN converged restart, or ``None``.

    **No cross-candidate fallback.**  The previous implementation fell back to
    ``sorted((run_dir / "master").rglob("MAS_RST.*"))[0]`` — an arbitrary restart
    belonging to whichever candidate sorted first.  A branch run on candidate A's
    deck against candidate B's equilibrium restart completes normally and prints a
    perfectly plausible MTC/SDM number *for a core that was never evaluated*: a
    silent wrong answer in the one place the pipeline exists to be trustworthy.
    An unresolvable restart is now a SKIP with a reason (see
    :func:`candidates_from_delivery`), never a substitution.
    """

    for cand in sorted(folder.glob("MAS_RST.*")):
        if cand.is_file():
            return cand
    return None


# --------------------------------------------------------------------------- #
# per-candidate branch-asset provenance index (sdm_mtc_targets.jsonl)
# --------------------------------------------------------------------------- #
def record_target(
    run_dir: str | Path,
    record_id: str,
    *,
    deck_path: str | Path | None,
    restart_path: str | Path | None,
    tag: str = "",
    note: str = "",
) -> Path:
    """Append one candidate's branch assets to the run's provenance index.

    Written by the campaign as each candidate converges, because that is the ONLY
    moment the equilibrium chain's final deck + generated restart are both known
    and still on disk (the purge deletes intermediate cycles immediately, and a
    ``keep_success=False`` verifier deletes the final one too — in which case the
    entry records ``restart=None`` and the candidate is honestly unverifiable
    rather than silently mispaired).
    """

    path = Path(run_dir) / TARGETS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "record_id": str(record_id),
        "tag": str(tag or record_id[:12]),
        "deck": str(deck_path) if deck_path else None,
        "restart": str(restart_path) if restart_path else None,
        "note": note,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_targets(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """``{record_id: {deck, restart, tag, ...}}`` from the run's index (last wins)."""

    path = Path(run_dir) / TARGETS_NAME
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("record_id")
            if rid:
                out[str(rid)] = row
    return out


# --------------------------------------------------------------------------- #
# D9 pre-delivery gate — post-verify the campaign's flat feasible top-K
# --------------------------------------------------------------------------- #
@dataclass
class PostVerifySummary:
    """Outcome of one :func:`post_verify_delivery` pass (the D9 gate).

    ``master_calls`` is the licensing budget actually spent — the number the
    campaign adds to its status/report **separately** from the search budget, so
    neither figure can absorb the other.
    """

    results: list[SdmMtcResult] = field(default_factory=list)
    #: ``[{record_id, reason}]`` — selected candidates that could NOT be verified.
    skipped: list[dict[str, str]] = field(default_factory=list)
    master_calls: int = 0
    n_selected: int = 0
    report_only: bool = True
    limits: dict[str, Any] = field(default_factory=dict)
    table_path: str | None = None

    @property
    def violators(self) -> list[str]:
        """record_ids that explicitly violated a GATED limit."""
        return [r.record_id for r in self.results if r.violates]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_selected": self.n_selected,
            "n_verified": len(self.results),
            "master_calls": self.master_calls,
            "report_only": self.report_only,
            "violators": self.violators,
            "skipped": list(self.skipped),
            "limits": dict(self.limits),
            "table": self.table_path,
            "results": [r.to_dict() for r in self.results],
        }


def candidates_from_delivery(
    run_dir: str | Path,
    delivery: Mapping[str, Any] | None,
    top_k: int,
) -> tuple[list[CandidateRef], list[dict[str, str]]]:
    """Resolve the delivery ranking's top-K into verifiable candidates.

    ``delivery`` is the campaign's ``delivery.json`` payload (program §2.2 / D2):
    its ``ranked`` list is exactly D9's target set — candidates inside the cell's
    FLAT BAND that are feasible **excluding F_r** (the ``flat_power`` feasibility
    set gates F_q / CBC / |AO| / pin-BU, with F_r only a 1.70 safety gate), ordered
    by licensing headroom.  Loading it from ``run_dir`` is the fallback when the
    caller has no payload in hand.

    Returns ``(candidates, skipped)``.  A ranked entry with no resolvable deck +
    OWN converged restart lands in ``skipped`` with a reason — the alternative
    (borrowing another candidate's restart) is a silently wrong licensing number.
    """

    run_dir = Path(run_dir)
    if delivery is None:
        try:
            delivery = json.loads((run_dir / "delivery.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            delivery = {}
    ranked = list((delivery or {}).get("ranked") or [])[: max(int(top_k), 0)]
    targets = load_targets(run_dir)

    candidates: list[CandidateRef] = []
    skipped: list[dict[str, str]] = []
    for entry in ranked:
        rid = str(entry.get("record_id") or "")
        if not rid:
            skipped.append({"record_id": "", "reason": "delivery entry has no record_id"})
            continue
        target = targets.get(rid)
        if target is None:
            skipped.append({
                "record_id": rid,
                "reason": (f"no entry in {TARGETS_NAME}: this campaign did not record "
                           "the candidate's converged deck/restart, so its branches "
                           "cannot be built from its OWN equilibrium state"),
            })
            continue
        deck = target.get("deck")
        restart = target.get("restart")
        if not deck or not restart:
            skipped.append({
                "record_id": rid,
                "reason": (f"incomplete branch assets (deck={bool(deck)}, "
                           f"restart={bool(restart)}); "
                           + str(target.get("note") or "")).strip(),
            })
            continue
        deck_path, restart_path = Path(deck), Path(restart)
        if not deck_path.is_file() or not restart_path.is_file():
            skipped.append({
                "record_id": rid,
                "reason": ("branch assets no longer on disk (purged?): "
                           f"deck={deck_path.name} exists={deck_path.is_file()}, "
                           f"restart={restart_path.name} exists={restart_path.is_file()}"),
            })
            continue
        try:
            candidates.append(CandidateRef.from_paths(
                record_id=rid,
                tag=str(target.get("tag") or rid[:12]),
                deck_path=deck_path,
                restart_path=restart_path,
                metrics={k: entry.get(k) for k in
                         ("node_peak", "map_cov", "f_r", "compliance_margin",
                          "peak_percentile")},
            ))
        except OSError as exc:
            skipped.append({"record_id": rid, "reason": f"deck unreadable: {exc}"})
    return candidates, skipped


def post_verify_delivery(
    run_dir: str | Path,
    delivery: Mapping[str, Any] | None,
    constraints: Any,
    *,
    sdm_mtc_cfg: Any = None,
    master_cfg: Mapping[str, Any] | None = None,
    executor: BranchExecutor | None = None,
    rod_model: RodModel | None = None,
    sidecar_path: str | Path | None = DEFAULT_SIDECAR,
    write_table: bool = True,
) -> PostVerifySummary:
    """Run the D9 SDM/MTC gate on a campaign's top-K flat feasible candidates.

    Driven entirely by ``[constraints]``: nothing runs unless an axis is enabled,
    and an enabled axis without a user limit runs REPORT-ONLY (measured, recorded,
    never a violator).  ``rod_model`` supplies the full-core ``%ROD_CFG`` /
    ``%ROD_MAP`` an SDM branch requires; without it SDM stays INCONCLUSIVE and
    says so rather than reporting a vacuous pass.

    Never raises: a licensing-stage failure must not destroy a finished campaign's
    results.  Every failure mode is a row in the returned summary.
    """

    run_dir = Path(run_dir)
    limits = LicensingLimits.from_constraints(constraints, sdm_mtc_cfg)
    top_k = int(getattr(constraints, "post_verify_top_k", 0) or 0)
    summary = PostVerifySummary(limits=limits.to_dict())

    enabled_mtc = bool(getattr(constraints, "mtc_enable", False))
    enabled_sdm = bool(getattr(constraints, "sdm_enable", False))
    if top_k <= 0 or not (enabled_mtc or enabled_sdm):
        return summary

    candidates, skipped = candidates_from_delivery(run_dir, delivery, top_k)
    summary.skipped = skipped
    summary.n_selected = len(candidates) + len(skipped)
    if not candidates:
        return summary

    mtc_params = BranchParams(
        mtc_delta_c=float(getattr(sdm_mtc_cfg, "mtc_delta_c", 5.0)),
        mtc_output_units=str(getattr(sdm_mtc_cfg, "mtc_output_units", "pcm_per_c")),
    )
    sdm_params = None
    if enabled_sdm and rod_model is not None:
        sdm_params = BranchParams(
            scram_banks=tuple(getattr(sdm_mtc_cfg, "scram_banks", None)
                              or ("R1", "R2", "R3", "R4", "R5")),
            stuck_candidate_banks=tuple(
                getattr(sdm_mtc_cfg, "stuck_candidate_banks", None)
                or ("R1", "R2", "R3", "R4", "R5")),
            rod_model=rod_model,
        )
    elif enabled_sdm:
        summary.skipped.append({
            "record_id": "*",
            "reason": ("SDM enabled but no full-core rod model available; the "
                       "quarter-core campaign deck carries no %ROD_CFG/%ROD_MAP "
                       "(see module docs) — SDM reported INCONCLUSIVE, not passed"),
        })

    work_root = run_dir / "sdm_mtc"
    try:
        results = run_post_verification(
            candidates, limits, master_cfg, work_root,
            mtc_params=mtc_params, sdm_params=sdm_params, executor=executor,
            sidecar_path=sidecar_path, run_mtc=enabled_mtc,
        )
    except Exception as exc:  # noqa: BLE001 — the gate never sinks a finished run
        summary.skipped.append({"record_id": "*", "reason": f"{type(exc).__name__}: {exc}"})
        return summary

    summary.results = list(results)
    summary.master_calls = sum(r.master_calls for r in results)
    summary.report_only = not (limits.mtc_gated or limits.sdm_gated)
    if write_table and results:
        try:
            summary.table_path = str(write_verdict_table(results, work_root, limits))
        except OSError:
            summary.table_path = None
    try:
        _atomic_write_json(run_dir / "sdm_mtc.json", summary.as_dict())
    except OSError:
        pass
    return summary


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def post_verify_topk(
    run_dir: str | Path,
    criteria_spec: Mapping[str, Any] | None,
    master_cfg: Mapping[str, Any] | None,
    *,
    executor: BranchExecutor | None = None,
    sidecar_path: str | Path | None = DEFAULT_SIDECAR,
) -> list[SdmMtcResult]:
    """Campaign/report integration hook (additive — NOT wired into the live loop).

    Selects the top-K feasible candidates of ``run_dir`` and licence-verifies
    them.  ``criteria_spec`` keys (all optional): ``top_k`` (default 5),
    ``mtc_limit`` (most-positive MTC), ``sdm_limit`` (minimum SDM), ``mtc_delta_c``,
    ``mtc_output_units``, ``rod_model`` (a :class:`RodModel` for SDM), ``states``.
    The curriculum agent owns ``campaign.py``; this function is merely exported
    for campaign/report to call later.
    """

    spec = dict(criteria_spec or {})
    top_k = int(spec.get("top_k", 5))
    limits = LicensingLimits.from_overrides(
        mtc_limit=spec.get("mtc_limit"), sdm_limit=spec.get("sdm_limit")
    )
    mtc_params = BranchParams(
        states=tuple(spec.get("mtc_states", ())),
        mtc_delta_c=float(spec.get("mtc_delta_c", 5.0)),
        mtc_output_units=str(spec.get("mtc_output_units", "pcm_per_c")),
    )
    rod_model = spec.get("rod_model")
    sdm_params = None
    if rod_model is not None:
        sdm_params = BranchParams(
            states=tuple(spec.get("sdm_states", ())),
            scram_banks=tuple(spec.get("scram_banks", ("R1", "R2", "R3", "R4", "R5"))),
            stuck_candidate_banks=tuple(
                spec.get("stuck_candidate_banks", ("R1", "R2", "R3", "R4", "R5"))
            ),
            rod_model=rod_model,
        )
    candidates = select_topk_feasible(run_dir, top_k)
    work_root = Path(run_dir) / "sdm_mtc"
    return run_post_verification(
        candidates, limits, master_cfg, work_root,
        mtc_params=mtc_params, sdm_params=sdm_params, executor=executor,
        sidecar_path=sidecar_path,
    )


# --------------------------------------------------------------------------- #
# verdict table (written into the run's report dir)
# --------------------------------------------------------------------------- #
def write_verdict_table(results: Sequence[SdmMtcResult], out_dir: str | Path,
                        limits: LicensingLimits) -> Path:
    """Write ``sdm_mtc_report.md`` (+ ``.csv``) verdict table into a run's report dir."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "sdm_mtc_report.md"
    csv = out_dir / "sdm_mtc_summary.csv"

    def _gate_note(gated: bool) -> str:
        return "GATED" if gated else "REPORT-ONLY (no user limit set)"

    lines = [
        f"# SDM / MTC post-verification — {out_dir.name}",
        "",
        f"- MTC window: **{limits.mtc_min_pcm_per_c:g} .. {limits.mtc_max_pcm_per_c:g} "
        f"pcm/°C** (max = most-positive allowed) — {_gate_note(limits.mtc_gated)}  ",
        f"- SDM required: **≥ {limits.sdm_required_pcm:g} pcm** "
        f"(= CEA allowance {limits.cea_allowance_pcm:g} + uncertainty "
        f"{limits.net_worth_uncertainty_pcm:g}) — {_gate_note(limits.sdm_gated)}  ",
        f"- limits source: {limits.limits_source}  ",
        f"- MASTER branch calls spent here: "
        f"**{sum(r.master_calls for r in results)}** (licensing budget, counted "
        f"separately from the search budget)  ",
        "",
        "| tag | record_id | MTC [pcm/°C] | MTC | SDM margin [pcm] | SDM | verdict | note |",
        "|---|---|---:|:--:|---:|:--:|:--:|---|",
    ]
    csv_rows = [
        "tag,record_id,mtc_pcm_per_c,mtc_pass,sdm_margin_pcm,sdm_pass,verdict,status,note"
    ]
    for r in results:
        mtc_v = "n/a" if not r.mtc or r.mtc.value_pcm_per_c is None else f"{r.mtc.value_pcm_per_c:.2f}"
        sdm_m = "n/a" if not r.sdm else f"{r.sdm.margin_pcm:.1f}"
        mtc_p = _verdict_cell(r.pass_mtc)
        sdm_p = _verdict_cell(r.pass_sdm)
        verdict = r.verdict
        note = "; ".join(
            filter(None, [
                r.failure,
                r.mtc.warning if r.mtc else "",
                r.sdm.warning if r.sdm else "",
            ])
        )
        lines.append(
            f"| {r.tag} | `{r.record_id[:12]}…` | {mtc_v} | {mtc_p} | {sdm_m} | "
            f"{sdm_p} | {verdict} | {note} |"
        )
        csv_rows.append(
            f"{r.tag},{r.record_id},{mtc_v},{r.pass_mtc},{sdm_m},{r.pass_sdm},"
            f"{verdict},{r.status},\"{note}\""
        )
    n_violate = sum(1 for r in results if r.violates)
    lines += [
        "",
        f"_{len(results)} candidate(s) verified; "
        f"{sum(1 for r in results if r.passed)} PASS, {n_violate} VIOLATOR(S)._",
    ]
    if not (limits.mtc_gated or limits.sdm_gated):
        lines += [
            "",
            "> **Report-only run.** No user limit is set in `[constraints]`, so no "
            "candidate can be a violator here. The absence of a FAIL is a "
            "measurement, not a clearance — set `mtc_max_pcm_per_c` / "
            "`sdm_required_pcm` to turn these numbers into a gate.",
        ]
    lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    csv.write_text("\n".join(csv_rows) + "\n", encoding="utf-8-sig")
    return md


def _verdict_cell(p: bool | None) -> str:
    return "PASS" if p is True else ("FAIL" if p is False else "—")
