"""Typed configuration for the lpopt campaign TOML ``.inp`` deck.

The deck is parsed with the standard-library :mod:`tomllib` (Python >= 3.11) into
frozen-ish dataclasses.  Unknown keys are a *hard error* (typo protection): the
loader collects every key that does not map to a known field and raises with the
full list, so a mistyped ``[remote] hosst`` fails loudly instead of being
silently ignored.

Only the milestone-M0 sections are modelled here: ``[flow]``, ``[remote]``,
``[master]``, ``[verify]``, ``[data]``, ``[case]``.  Later milestones extend this
module (``[produce]``, ``[search.trust_region]``, ...).
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised for malformed decks or unknown keys/sections."""


@dataclass
class FlowConfig:
    """``[flow]`` — run-level metadata."""

    title: str = "lpopt"
    output_root: str = "runs"
    random_seed: int = 0


@dataclass
class RemoteConfig:
    """``[remote]`` — gpu2-6000 remote training host (defaults per plan 4.7)."""

    host: str = "HOST_238"
    user: str = "USER"
    port: int = 8022
    workdir: str = "~/lpopt_ws"
    gpu: str | int = "auto"
    env: str = "~/lpopt_ws/venv"
    tmux_prefix: str = "lpopt"


@dataclass
class MasterConfig:
    """``[master]`` — local MASTER executable and equilibrium run knobs.

    ``tolerances`` is an optional ``[master.tolerances]`` sub-table mapping any of
    the five equilibrium metric keys (``cyclen cbc_max f_q f_r ao``) to an
    absolute successive-cycle tolerance (``None`` disables that axis).  When
    omitted the vendor :data:`EquilibriumTolerances` defaults apply.  ``cache_dir``
    overrides the per-run MASTER metric cache location (default: under the run
    directory).  ``keep_success`` retains converged work dirs for diagnosis.

    Core policy (shared with ``[produce]``): ``workers = 0`` means *auto* — the
    :class:`~lpopt.search.verify.WaveVerifier` fills the assignable core pool.  A
    positive ``workers`` caps the worker count.  ``use_all_cores`` (default
    ``False`` for OPTIMIZE/user_criteria campaigns) selects the P-cores-only pool
    (legacy) vs the full P-then-E pool; ``host_reserve`` is how many logical cores
    are held back for the host process when ``use_all_cores`` is on.
    """

    executable: str | None = None
    workers: int = 0                         # 0 = auto (fill the core pool)
    timeout: int = 3600
    #: Per-MASTER wall cap on the ``design bootstrap`` path ONLY (cy1 + every
    #: reload cycle of ``make_band_restart``).  ``timeout`` above stays the
    #: campaign/produce cap and is NOT changed by this key.  A healthy bootstrap
    #: cycle is 17-24 s; the 2026-09-03 S6 failure burned 3,600 s per divergence
    #: because ``timeout`` was the only defence.  The NaN watchdog now kills a
    #: divergence in ~10 s, so this is the backstop for a genuine hang.
    bootstrap_timeout_s: float = 900.0
    max_cycles: int = 16
    consecutive: int = 2
    tolerances: dict[str, float | None] | None = None
    cache_dir: str | None = None
    keep_success: bool = False
    #: OPTIMIZE campaigns default to the 8 P-cores; flip to spread across all
    #: logical cores (P-cores first, then E-cores).
    use_all_cores: bool = False
    #: logical cores held back for the host process when ``use_all_cores`` is on.
    host_reserve: int = 1


@dataclass
class VerifyConfig:
    """``[verify]`` — MASTER package root (FEASIBLE_PACKAGE layout)."""

    package_root: str | None = None
    #: Harvest the converged EDIT5 assembly maps (boc/eoc power+burnup+kinf) from
    #: each candidate's MAS_SUM into the campaign store's ``maps.npz`` (keyed by
    #: record_id).  Forces the verifier's ``keep_success`` so the final cycle dir
    #: survives to the harvest.  Default False = byte-identical (no map write, no
    #: retained dirs).  The fr_boundary / flat_power kits set it True so the node-
    #: power-distribution learning has campaign-label maps (forensic 20260723: the
    #: only maps in the store were Dataset-A f121 — the map head never saw the
    #: boundary region).
    harvest_maps: bool = False


@dataclass
class DataConfig:
    """``[data]`` — extraction source path lists (all optional)."""

    sources: list[str] = field(default_factory=list)
    lp_cache: list[str] = field(default_factory=list)
    lp_case_decks: list[str] = field(default_factory=list)
    eqlp_ws: list[str] = field(default_factory=list)
    ga_manifests: list[str] = field(default_factory=list)
    ga_event_logs: list[str] = field(default_factory=list)


@dataclass
class CaseConfig:
    """``[case]`` — search mode / external decision variables (plan sec. 6.2).

    ``mode`` is one of ``fixed`` | ``feed_range`` | ``free`` | ``user_criteria``.
    ``fixed`` and ``user_criteria`` (plan sec. 12.5) are implemented end-to-end;
    ``feed_range`` / ``free`` parse-validate here but the campaign raises
    ``NotImplementedError`` at start (a clear deferred-milestone message).
    ``feed_range`` / ``pairs`` / ``e_core_range`` carry the extra decision
    variables the later modes will consume.  ``user_criteria`` treats the fuel
    PAIR itself as the outer decision variable (chosen from the ``[criteria]``
    e_core-reachable universe), so it needs no ``pair`` — only the fixed ``feed``.
    """

    mode: str = "fixed"
    pair: str | None = None
    feed: int = 121
    feed_range: list[int] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    e_core_range: list[float] = field(default_factory=list)

    def validate(self) -> None:
        if self.mode not in ("fixed", "feed_range", "free", "user_criteria"):
            raise ConfigError(
                f"[case] mode {self.mode!r} invalid; expected "
                "fixed|feed_range|free|user_criteria"
            )
        # user_criteria picks the pair from the [criteria] universe -> no pair.
        if self.mode in ("fixed", "feed_range") and not self.pair:
            raise ConfigError(f"[case] mode {self.mode!r} requires a pair")
        if (int(self.feed) - 1) % 4 != 0:
            raise ConfigError(f"[case] feed {self.feed} is not on the 1+4N grid")


@dataclass
class TrustRegionConfig:
    """``[search.trust_region]`` — support-grid gate (plan sec. 6.2)."""

    enabled: bool = True
    feed_step: int = 4                       # one reachable feed step = 4 assemblies
    e_core_band: float = 0.10                # +/- e_core band [w/o]
    n_min: int = 50                          # min labels per (feed, e_core) bin
    promote_after: int = 16                  # verified labels that grow a bin
    frontier_sigma_inflation: float = 1.5    # sigma multiplier on frontier bins
    frontier_slots_per_wave: int = 1


@dataclass
class LocalSearchConfig:
    """``[search.local_search]`` — first-improvement refinement (plan sec. 4.6)."""

    top_m: int = 256
    neighbors: int = 200
    depth: int = 3
    max_predictions: int = 40000
    n_moves: int = 1


@dataclass
class SearchConfig:
    """``[search]`` — candidate-pool construction knobs (plan sec. 4.6)."""

    pool_size: int = 20000
    pool_cap: int = 100000
    elite_frac: float = 0.60
    guided_frac: float = 0.30
    diversity_frac: float = 0.10
    beam_width: int = 4
    completions_per_prefix: int = 8
    n_moves_early: int = 2                    # small trust-region moves (early waves)
    n_moves_late: int = 5
    elite_top_k: int = 32
    #: near-miss parent inclusion: verified this-campaign rows with F_r <= this
    #: bound join the elite-mutation parent set with an n_moves=1 (small trust-
    #: region) bias, so the pool tightly explores the almost-feasible boundary
    #: even before a fully-feasible label exists (plan sec. 4.6).  0 disables.
    #:
    #: The F_r RULE applies only where F_r is the objective/constraint axis.  A
    #: ``flat_power`` campaign seeds the same arm from its OWN objective (the
    #: ``near_miss_top_k`` flattest labelled rows) — seeding a flatness search
    #: with low-F_r parents is the elite-seeding leak program §10 STOPs.  ``0``
    #: still disables the arm in every mode.
    near_miss_f_r: float = 1.60
    #: flatness-objective near-miss parent count (``flat_power``): the K flattest
    #: verified this-campaign rows carrying a ``node_peak`` label seed the tight
    #: local-search arm.  A COUNT rather than a bound because the flatness scalar
    #: is normalized per cell — there is no cross-cell constant to compare to.
    near_miss_top_k: int = 8
    #: DONOR case ids whose converged store rows also seed the elite-mutation
    #: parent set (``CampaignDriver._store_elites``), on top of the campaign's own
    #: ``[case] pair``.  Empty (the default) leaves every existing deck's parent
    #: set, rng draw sequence and pool byte-identical.
    #:
    #: Exists for the 3-fresh-type cold start (data/reports/tripletype_design_
    #: 20260817.md §3.3): a graded ``A_B_C`` case matches ZERO store rows by
    #: ``case_pair``, so without this knob its elite pool is empty and the
    #: ``graded_morph`` operator — which needs a 2-type parent to re-label — is
    #: never reached.  Naming the parent PAIR here hands the campaign its own
    #: optimized 2-type ancestors, which ``mutate(..., batches=<3 types>)`` then
    #: morphs into 3-type children.
    #:
    #: The donor rows are real MASTER labels of a DIFFERENT case: they are used as
    #: mutation PARENTS only.  They are not added to the campaign's holdout, are
    #: not counted as this case's verified rows, and are never reported as in-cell
    #: truth.
    elite_seed_cases: list[str] = field(default_factory=list)
    #: Require every candidate to FEED every member of the case's fresh-type
    #: alphabet (``CaseContext.require_all_batches``).  Default off => existing
    #: decks are byte-identical, and it is a no-op for a 2-type case anyway.
    #:
    #: Turn it on for a GRADED (3+ type) campaign whose budget must buy graded
    #: cores.  Without it the pool legally contains boards that fed only two of
    #: the three types; those are the same physical cores the 2-type campaign at
    #: the same cell already measured, and an out-of-distribution model will
    #: happily rank them first.
    require_all_fresh_types: bool = False
    #: dry-run pool cap (CPU-friendly StubEvaluator acceptance; live uses pool_size).
    dry_run_pool_size: int = 400
    trust_region: TrustRegionConfig = field(default_factory=TrustRegionConfig)
    local_search: LocalSearchConfig = field(default_factory=LocalSearchConfig)


@dataclass
class AcquisitionConfig:
    """``[acquisition]`` — budget / wave composition / gates (plan sec. 4.6)."""

    budget: int = 100
    wave_size: int = 8
    exploit: int = 5
    explore: int = 2
    control: int = 1
    reserve: int = 4
    n_waves: int = 12
    tau0: float = 0.30
    hamming_min: int = 4
    #: Campaign objective selector (plan sec. 1-4 + user directive 2026-07-21).
    #: ``"target_cycle"`` (default) — the historical behaviour: pin the verified
    #: cyclen to ``cycle_target_efpd`` within ``±cycle_tolerance_efpd`` and minimize
    #: CBC/F_r inside the window, with F_r a HARD constraint at ``f_r_limit``.
    #: ``"max_cycle_min_fr"`` — MAXIMIZE cyclen and MINIMIZE F_r subject to the
    #: remaining constraints (F_q, CBC, |AO|); F_r is NO LONGER gated, it becomes
    #: the secondary objective.
    #: ``"min_fr_max_cycle"`` — MINIMIZE F_r (primary) with cyclen maximization the
    #: secondary tie-break, and F_r <= f_r_limit REJOINS the hard constraint set
    #: (F_r, F_q, CBC, |AO| all gated).  A deck that never sets ``objective`` is
    #: byte-identical to the pre-change target_cycle campaign.
    #: ``"min_fxy"`` — MINIMIZE **F_xy** (MASTER ``FXYP``, the pin PLANAR peaking
    #: factor) as the PRIMARY objective with cyclen the secondary tie-break, under
    #: the hard limit ``f_xy_limit`` (1.65, user decision 2026-08-29).  F_r STAYS a
    #: hard constraint at ``f_r_limit`` (1.55) — measured cores exist that pass one
    #: axis and fail the other (design fxy_switch_20260829 §1.3), so F_r is not a
    #: sufficient proxy and dropping it would silently RELAX the feasible set.
    objective: str = "target_cycle"
    #: ``max_cycle_min_fr`` scalarization weight λ [EFPD per unit F_r]: the exploit
    #: score is ``cyclen_LCB − λ·F_r_UCB`` (both risk-adjusted at ``risk_z``).
    #: Default 100.0 is calibrated so a ~10 EFPD cyclen gain trades against a ~0.1
    #: F_r reduction (λ·0.1 = 10 EFPD): cyclen dominates, F_r breaks ties and pulls
    #: the frontier toward lower peaking.  Ignored outside max_cycle_min_fr.
    mcmf_lambda: float = 100.0
    #: ``min_fr_max_cycle`` scalarization weight λ_Fr [EFPD per unit F_r]: the
    #: exploit score is ``cyclen_LCB − λ_Fr·F_r_UCB``.  Default 1000.0 sizes F_r to
    #: STRICTLY dominate cyclen (λ_Fr·0.01 = 10 EFPD, so a 0.01 F_r reduction is
    #: worth a full 10 EFPD; the cell's ~30 EFPD cyclen spread only breaks ties among
    #: near-equal-F_r candidates).  Ignored outside min_fr_max_cycle.
    minfr_lambda: float = 1000.0
    #: ``min_fr_max_cycle`` predicted max-pin-burnup hard gate [GWd/MTU].  Default
    #: **78.0**, deliberately BELOW the LEU+ 80 the other pin-gated modes use: the
    #: extra 2.0 is MODEL margin from the in-cell validation of the s1g pin head at
    #: E1_E2/f109 (n=33, MAE 1.84, bias −1.39 — it UNDER-predicts), so a core gated
    #: at 78.0 is expected at ~79.4 after the bias correction, still inside 80.
    #: Set to 80.0 to gate on the licensing number directly.  Added 2026-08-17:
    #: this objective previously gated NOTHING on pin burnup while ``min_fuel_cost``
    #: / ``fr_boundary`` / ``flat_power`` all did, and both closed min_fr campaigns
    #: reported 100% of their feasible cores over 80 on the validated prediction
    #: (data/reports/fpcamp_E1E2_f109_results_20260817.md §7).
    #:
    #: **OBSERVABLE CONFIRMED 2026-08-20** (user ruling, `data/reports/
    #: pinbu_definition_20260820.md`): the licensing quantity is the **pin axial
    #: peak** — i.e. exactly what ``max_pin_burnup`` measures/predicts (the 3-D
    #: pin NODAL peak of ``MAS_PPI``'s ``BPIN``), and the limit is **80 GWd/tU**
    #: on that axis.  Rod-average (``max_rod_avg_burnup``) is a SECONDARY
    #: observable only.  So this gate and the limit live on the SAME axis and
    #: 78.0 = 80 − 2.0 is a genuine, correctly-scaled 2.0 GWd/tU model margin.
    #: (The pin-burnup audit's E16 — "a node prediction gated against a
    #: rod-average limit" — is WITHDRAWN; see pinbu_audit_20260820.md §8.4.)
    #:
    #: **OOD CAVEAT — the 2.0 does not cover everything.**  The pin head trained
    #: on ZERO feed-113 labels and ZERO F_r < 1.55 labels, so every delivery-region
    #: pin prediction is an extrapolation.  Measured: pooled bias −0.32 (41 cores,
    #: but 32/41 in-sample), and in a basin with no training core within Hamming 30
    #: the SAME head under-predicted by **−5.93 GWd/tU** (MAE 5.93, n=5) — about 3x
    #: this margin (pinbu_audit_20260820.md §4.3-4.4).  Campaigns aiming at
    #: DELIVERY must MEASURE their winners (``pinbu_wave.py``; the
    #: ``keep_success=true`` deck ``pinbu_wave_keep_199.inp`` yields the node and
    #: the rod-average peak in one run), not ship on the prediction alone.
    #: DEFAULT DELIBERATELY UNCHANGED at 78.0.
    minfr_pin_bu_limit: float = 78.0
    #: ``min_fxy`` scalarization weight λ_Fxy [EFPD per unit F_xy]: the exploit
    #: score is ``cyclen_LCB − λ_Fxy·F_xy_UCB``, the SAME λ structure
    #: ``minfr_lambda`` gives ``min_fr_max_cycle``.  Default 1000.0 sizes F_xy to
    #: strictly dominate cyclen (a 0.01 F_xy reduction is worth a full 10 EFPD).
    #: The measured within-cell F_xy spread is the same order as F_r's, so the
    #: F_r default carries over unchanged (design §3.5.1).  Ignored outside min_fxy.
    minfxy_lambda: float = 1000.0
    #: ``min_fxy`` predicted max-pin-burnup hard gate [GWd/MTU] — mirrors
    #: ``minfr_pin_bu_limit`` (78.0 = LEU+ 80 minus the 2.0 model margin of the
    #: pin head).  Ignored outside min_fxy.
    minfxy_pin_bu_limit: float = 78.0
    #: ``min_fxy`` OPTIONAL cyclen acceptance band [EFPD] (``None`` = no band, the
    #: default: cyclen is then a pure secondary tie-break, exactly as in
    #: ``min_fr_max_cycle``).  Set BOTH edges to make cyclen a hard constraint in
    #: the ``min_fuel_cost`` shape.
    minfxy_cyclen_lo: float | None = None
    minfxy_cyclen_hi: float | None = None
    #: ``min_fxy`` cyclen-band penalty normalizer [EFPD]; inert when no band is set.
    minfxy_cyclen_width: float = 10.0
    #: ``min_fuel_cost`` — MINIMIZE the fresh fuel-economics metric FE (total fresh
    #: U-235 charge = Σ_fresh u_mass×enrichment) as the PRIMARY objective, F_r the
    #: secondary tie-break, subject to ALL SIX hard constraints (cyclen band both
    #: edges [``fuelcost_cyclen_lo``, ``fuelcost_cyclen_hi``], F_r ≤ f_r_limit,
    #: F_q ≤ f_q_limit, CBC ≤ cbc_limit, |AO| ≤ ao_abs_limit, predicted
    #: max_pin_burnup ≤ ``fuelcost_pin_bu_limit``).  Ignored outside min_fuel_cost.
    #: ``fuelcost_lambda_fr`` [FE units per unit F_r] sizes F_r as a SUBORDINATE
    #: tie-break: FE is position-invariant (constant within a cell) so any positive
    #: λ_Fr fully orders the within-cell F_r ties, while the cross-cell FE gaps
    #: (≥ one feed step) keep FE dominant.  Default 20.0 (a 0.1 F_r drop = 2 ga80
    #: [pos·w/o] FE units, far below one feed step ≈ 22 units).
    fuelcost_lambda_fr: float = 20.0
    #: min_fuel_cost cyclen acceptance band [EFPD]: 625 ± 10 (user LEU+ directive).
    fuelcost_cyclen_lo: float = 615.0
    fuelcost_cyclen_hi: float = 635.0
    #: min_fuel_cost cyclen-band penalty normalizer [EFPD] (graded part of the
    #: already-dominant out-of-band penalty; ranking within-band is pure FE/F_r).
    fuelcost_cyclen_width: float = 10.0
    #: min_fuel_cost predicted max-pin-burnup hard limit [GWd/MTU] — LEU+ 80 (user
    #: explicit).  Screened on the physics point estimate; MASTER adjudicates final.
    fuelcost_pin_bu_limit: float = 80.0
    #: ``fr_boundary`` — MINIMIZE F_r (pure objective, NOT gated) while biasing the
    #: search toward the F_r=1.55 licensing boundary via a MID-TIER band-shaping
    #: penalty on the predicted F_r MEAN.  Subject to CBC/F_q/|AO| (gated) + a
    #: None-tolerant predicted pin-BU screen; cyclen is recorded but NEVER gated.
    #: The band ``[fr_boundary_band_lo, fr_boundary_band_hi]`` is an OOD/plausibility
    #: window: a predicted-F_r mean outside it is pushed below the whole in-band
    #: range (coefficient 100 in :func:`score_fr_boundary`).  Ignored elsewhere.
    fr_boundary_band_lo: float = 1.45
    fr_boundary_band_hi: float = 1.70
    #: fr_boundary predicted max-pin-burnup hard screen [GWd/MTU] — LEU+ 80.
    fr_boundary_pin_bu_limit: float = 80.0
    #: fr_boundary OUTER roster (decision D5).  True (default) restricts the
    #: frontier race to the 16 cells program §5 shows empirically hold F_r ≤ 1.55
    #: rows (e_core 5.0–5.5 × feed 117/121/125 minus 5.4_f117 / 5.5_f117), which
    #: is what makes fr_boundary a DESIGN-SPACE MAPPING mode rather than a
    #: core-loading objective.  False restores the original 24-cell roster.
    #: Ignored outside the fr_boundary outer race.
    fr_boundary_compliant_only: bool = True
    #: ``flat_power`` — FLATNESS-NATIVE (program 20260725 §1.2/§1.3).  The
    #: objective is ``-( node_peak/PEAK_SCALE + w_cov * map_cov/COV_SCALE )`` with
    #: ``node_peak`` the PRIMARY term (weight 1.0) and ``map_cov`` SECONDARY, both
    #: UCB-conservatized at ``risk_z``.  F_r is NOT in the objective: it is a pure
    #: SAFETY GATE at ``flatpower_fr_limit`` (1.70, decision D1 — held there, not
    #: relaxed, while the map head is optimistic).  Remaining hard set: F_q, CBC,
    #: |AO|, predicted max_pin_burnup ≤ ``flatpower_pin_bu_limit``; cyclen
    #: record-only.  Ignored outside flat_power.
    flatpower_fr_limit: float = 1.7
    #: ``flat_power`` F_xy SAFETY GATE [-] (design §3.5.3).  Binary veto in the
    #: same shape as ``flatpower_fr_limit``: it can reject a candidate, it never
    #: grades one.  Needed because ``node_peak`` (the flatness objective) is a
    #: BOC assembly radial peak whose measured correlation with MASTER's FXYP is
    #: only 0.74-0.85 — a flat pattern carries NO guarantee of F_xy compliance.
    #: ``0.0`` (or a non-finite value) disables the gate, restoring the previous
    #: flat_power behaviour exactly.
    flatpower_fxy_limit: float = 1.65
    flatpower_pin_bu_limit: float = 80.0
    #: SECONDARY-term weight ``w_cov`` (decision D4 default 0.5).  The declared
    #: ratio is only honest under per-cell normalization — see below.
    flatpower_w_cov: float = 0.5
    #: Per-cell normalization of the two flatness terms (decision D4 default ON).
    #: Fixed global constants make the REALIZED ``w_cov`` vary 0.25-1.25 across
    #: cells (measured spread 4.95x in within-cell SD units), so the declared 0.5
    #: would be a fiction; per-cell scales make it exact.  Set false ONLY to
    #: reproduce a fixed-constant run.  Scales come from
    #: ``<store_dir>/flat_scale.json`` (``python -m lpopt.tools.fit_flat_scale``);
    #: an absent artifact falls back to the measured global constants.
    flatpower_per_cell_scale: bool = True
    #: Overrides for the global fallback scales (0 = use the measured module
    #: defaults / the artifact).  Present so a deck can pin an experiment, NOT so
    #: someone can hand-tune the weight ratio.
    flatpower_peak_scale: float = 0.0
    flatpower_cov_scale: float = 0.0
    #: Minimum per-wave MAP HARVEST RATE (converged rows carrying a flatness
    #: label / converged rows) before the flat_power campaign HARD ABORTS
    #: (program §1.3).  The objective is defined on the map columns, so a wave
    #: that silently stops harvesting maps would otherwise leak into the
    #: early-stop path as "no improvement".  0 disables the abort.
    flatpower_min_map_harvest: float = 0.5
    #: ENGINEERING-RULE SOFT PENALTIES (``flat_power`` only; ALL DEFAULT 0.0 ==
    #: OFF == byte-identical to the previous objective).  Each weight multiplies
    #: one VALIDATED arrangement metric of :mod:`..search.rule_metrics` and the
    #: weighted sum is SUBTRACTED from the exploit score, in the SAME units as
    #: the normalized flatness scalar (so a weight of 0.01 against a typical
    #: RM1 spread of ~10 pairs shifts the score by ~0.1, comparable to a 0.1
    #: z-unit of node_peak).  This is a PREFERENCE, never a constraint: it
    #: reorders near-ties and can never veto a candidate — the source report's
    #: own lesson is that promoting a loading heuristic to a hard constraint
    #: truncates the search space (the McFLOP / "Ring-of-Fire" case).
    #:
    #: ``rm1``  fresh-fresh FACE adjacency, whole core (rule R-03; measured
    #:          within-cell rho vs node_peak +0.085, holdout residual rho +0.114).
    #: ``rm1i`` the same restricted to INBOARD pairs (both slots off the outer
    #:          ring) — the study's causal carrier, rho +0.235.  PREFER THIS ONE.
    #: ``rm2``  fresh-fresh DIAGONAL adjacency (rule R-04), rho +0.076.
    #: ``rm2i`` inboard diagonal, rho +0.172.
    #:
    #: RM3 / RM4 / RM5 / RM6 have NO knob on purpose: they were measured and are
    #: report-only (RM3 is -0.885 collinear with RM1; RM4 is null once
    #: conditioned on RM1i and a "prefer low RM4" arm made node_peak WORSE by
    #: +0.064; RM6 is noise; RM5 is same-map circular).
    flatpower_rule_penalty_rm1: float = 0.0
    flatpower_rule_penalty_rm1i: float = 0.0
    flatpower_rule_penalty_rm2: float = 0.0
    flatpower_rule_penalty_rm2i: float = 0.0
    #: ``policy_prior`` — the v1 learned MOVE-PROPOSAL policy at the elite-
    #: mutation step of :func:`lpopt.search.construct.build_pool`
    #: (``data/reports/policy_v1_results_20260815.md`` section 7).
    #:
    #: ``"off"`` (DEFAULT) loads no model, and the mutation loop draws from the
    #: rng in exactly the sequence it drew before this knob existed — a deck that
    #: never sets it is byte-identical to the pre-change campaign.
    #: ``"fr"`` ranks edits by P(improve F_r), ``"flat"`` by P(improve node_peak),
    #: ``"both"`` by the mean of the two.
    #:
    #: WHERE IT IS VALIDATED.  The policy passed its gate on 260624-family cells
    #: (held-out CELL precision@32 0.888, parent-blocked AUC 0.826) and did NOT
    #: pass on the ga80/paramA era the live program runs, where the free analytic
    #: power prior out-ranks it on pooled AUC (report sections 2 and 4).  Turning
    #: this on for a ga80/paramA cell is running an unvalidated prior, and the
    #: report's own next step is a paired A/B on a 260624 cell first.
    #:
    #: It is a RANKER, never a probability (report section 5: off-distribution
    #: ECE 0.111/0.200), and it only ever ranks edits WITHIN one parent — which
    #: is the readout that passed.  It does NOT replace the board scorer in
    #: ``_score_completions``: section 4 shows the two win at different things.
    #: ``"v1"`` is ``"both"`` under the version-explicit name; ``"v2"`` serves the
    #: v2 ensemble (:class:`lpopt.policy.scorer.MoveScorerV2`) from
    #: :attr:`policy_prior_model_dir_v2`; ``"shadow_v2"`` builds the pool exactly
    #: as ``"off"`` does — same rng sequence, same candidates — and only RECORDS
    #: v2's score for every elite child in the wave metadata
    #: (``policy_shadow_scores``).  Shadow is the arm that collects the review's
    #: prospective A/B data without letting an ungated policy touch selection.
    #: ``"v3"`` / ``"shadow_v3"`` are the same two roles for the v3 ensemble from
    #: :attr:`policy_prior_model_dir_v3`, ranking on its ``fxy`` head — the
    #: campaign objective itself rather than the F_r proxy the r1 wave rejected.
    policy_prior: str = "off"
    #: Checkpoint directory (5 ``cnn_seed*`` members + their ``meta.json``).
    policy_prior_model_dir: str = "data/models/policy_v1"
    #: Checkpoint directory for the v2 modes (``v2`` / ``shadow_v2``).  Separate
    #: from :attr:`policy_prior_model_dir` so an A/B deck can name both arms'
    #: checkpoints at once and switch arms by ``policy_prior`` alone.
    policy_prior_model_dir_v2: str = "data/models/policy_v2"
    #: Checkpoint directory for the v3 modes (``v3`` / ``shadow_v3``).  Separate
    #: for the same reason the v2 one is: the registered A/B/C names all three
    #: arms' checkpoints at once and switches arms by ``policy_prior`` alone.
    policy_prior_model_dir_v3: str = "data/models/policy_v3"
    #: FAIL-CLOSED switch (review section 6.12).  ``false`` (default) is research
    #: behaviour: a policy that will not load prints a WARNING, the elite arm
    #: falls back to unscored random mutation, and the wave metadata records
    #: ``policy_fallback = true`` so no readout can mistake the fallback for
    #: policy-on.  ``true`` is production behaviour: the deck asked for a policy,
    #: so a policy that will not load RAISES at pool construction rather than
    #: running a silent control arm under a policy-on label.
    policy_prior_strict: bool = False
    #: Softmax temperature over the selected head's probability.  The report's
    #: first safety rail: SAMPLE, never argmax — a hard argmax on a scorer with
    #: 0.650 era AUC collapses pool diversity.  0.25 leaves a ~11x sampling-odds
    #: ratio between a 0.8 and a 0.2 candidate and only ~3x between 0.7 and 0.4,
    #: so the policy reorders the neighbourhood without owning it.  Smaller is
    #: greedier; at 0 the pick degenerates to argmax and the rail is gone.
    policy_prior_temperature: float = 0.25
    #: Softmax temperature for the v2 modes.  NOT inheritable from the v1 value
    #: and that is the whole reason this field exists: v2's output is a
    #: normalized clipped expected improvement, not v1's probability, and its
    #: gate-fold p90−p10 spread is 0.189 (``fr``) against v1's 0.573.  At v1's
    #: τ = 0.25 the v2 softmax is nearly uniform and the prior would do almost
    #: nothing — a treatment arm that is its own control.  0.08 reproduces v1's
    #: ~10x sampling-odds ratio on v2's scale
    #: (``data/reports/policy_v2_results_20260817.md`` section 8, item 3).
    policy_prior_temperature_v2: float = 0.08
    #: Softmax temperature for the v3 modes.  **0.0 means NOT YET DERIVED and a
    #: selecting v3 arm refuses to build a pool with it** — the value is the one
    #: that reproduces v1's ~10x sampling-odds ratio on the v3 gate fold's own
    #: p90-p10 score spread, it is computed and written down BEFORE arm C
    #: launches, and re-deriving it after seeing an outcome is forbidden
    #: (``policy_v3_prereg_20260831.md`` §5d).  A default that happened to work
    #: would let an underived arm launch and be read as a treatment.
    policy_prior_temperature_v3: float = 0.0
    #: Edits proposed per scored parent expansion before the softmax pick
    #: (report section 7: "generate N ~ 16 mutations ... and admit the top m").
    policy_prior_candidates: int = 16
    #: Fraction of elite slots that stay UNSCORED random mutations — the report's
    #: second safety rail, so the pool cannot become policy-degenerate.  0.20
    #: keeps one elite child in five drawn exactly as the flag-off campaign draws
    #: it.  1.0 scores nothing; 0.0 removes the floor (not recommended).
    policy_prior_random_floor: float = 0.20
    #: torch intra-op threads for the scorer.  NOT cosmetic: measured on the
    #: 24-core workstation, letting torch take every core made a 16-candidate
    #: forward 20x SLOWER (8.6 s vs 0.37 s) and the feature encoder 5x slower,
    #: because the campaign process is already sharing the box with a MASTER
    #: queue.  0 leaves torch's default alone.
    policy_prior_threads: int = 4
    #: exploit tie-break band (plan sec. 4.6 refinement).  When two candidates'
    #: exploit scores fall within this band the risk-adjusted feasibility MARGIN
    #: LCB (min over constraints of ``(mu + risk_z*sigma - limit)/width``) breaks
    #: the tie toward the larger predicted margin — so among model look-alikes the
    #: tiny F_r differences and their uncertainty decide.  0 disables (pure
    #: exploit ranking, the previous behaviour).
    tie_epsilon: float = 0.1
    #: hard minimum Hamming distance an exploit pick must keep from EVERY already
    #: verified pattern (not just this wave's diversity set).  Blocks near-repeats
    #: the surrogate cannot discriminate from waste.  0 disables.
    exploit_verified_hamming: int = 2
    #: fine-tune boundary emphasis: this-campaign wave labels are oversampled this
    #: many times against the replay pool so the few fresh boundary labels move
    #: the discriminator (plan sec. 4.6).  1 = no emphasis (previous behaviour).
    finetune_new_weight: int = 4
    # objective (target_cycle) + constraints (plan sec. 1-4)
    cycle_target_efpd: float = 625.0
    cycle_tolerance_efpd: float = 2.0
    risk_z: float = 0.25
    f_r_limit: float = 1.55
    #: HARD limit on **F_xy** (MASTER ``FXYP`` — pin PLANAR peaking), user decision
    #: 2026-08-29.  It is the PRIMARY objective's gate under ``objective =
    #: "min_fxy"`` and the deliverability gate everywhere the objective screens
    #: F_xy.  It lives HERE and not in ``[constraints]``: that section is the
    #: SDM/MTC post-verification knob surface, while ``f_r_limit`` / ``cbc_limit`` /
    #: ``f_q_limit`` — the axes F_xy joins — have always lived in ``[acquisition]``.
    f_xy_limit: float = 1.65
    cbc_limit: float = 1550.0
    f_q_limit: float = 2.41
    ao_abs_limit: float = 0.30
    # -- SAFETY SHIELD (external review §6.5 P0-04 / §8.5) ------------------- #
    #: What the serve-time feature/geometry OOD guard
    #: (:mod:`lpopt.model.ood_guard`, surfaced by
    #: ``PosValCnnBackend.feature_ood_types``) is allowed to DO to the search.
    #:
    #: ``"warn"`` (DEFAULT) — today's behaviour EXACTLY: the guard is a warning
    #: surface, no candidate is demoted or dropped, and a deck that never sets
    #: this key produces a byte-identical wave.
    #: ``"escalate"`` — a candidate whose fresh types fall outside the training
    #: envelope loses its EXPLOIT score (demoted to ``-inf``, i.e. out of the
    #: exploit tier and out of the next wave's elite seeds) but stays eligible for
    #: the explore/control slots, which is where an off-manifold board belongs:
    #: the review's rule is "no surrogate-only exploit on an OOD point", not
    #: "never look at it".
    #: ``"reject"`` — the candidate is dropped from the pool entirely and the wave
    #: summary carries the count.
    #:
    #: The guard is the front line precisely BECAUSE ensemble σ is not: every
    #: member trained on the same manifold agrees while being jointly wrong
    #: (``ood_guard`` module docstring), so an OOD point's exploit score is
    #: confidently wrong rather than visibly uncertain.
    ood_policy: str = "warn"
    #: Use the split-conformal UPPER bound (:mod:`lpopt.model.conformal`, served by
    #: ``PosValCnnBackend.predict_interval``) as a HARD chance constraint
    #: ``U_c(x) <= L_c`` on the gated licensing axes, instead of leaving the
    #: mean+κ·σ UCB screen as the only bound (review §6.5 improvement 3).
    #: ``false`` (DEFAULT) keeps the conformal artifact report-only — the shipped
    #: behaviour.  ``true`` drops any candidate whose conformal upper bound on a
    #: gated axis exceeds that axis's limit and reports the per-axis count.
    #: An axis with NO fitted interval for a candidate is not screened by this
    #: gate (its mean+κ·σ UCB screen stands) and is reported as an unfit axis.
    conformal_gate: bool = False
    #: Miscoverage level the gate reads the interval at.  Must be one of the
    #: levels the artifact was FIT at (:data:`lpopt.model.conformal.DEFAULT_ALPHAS`
    #: — 0.10 for a 90 % interval, 0.32 for 68 %); asking for an unfitted level
    #: would silently serve a vacuous ``+inf`` bound.  Default 0.10, the primary
    #: level every fit selects its score type at.
    conformal_alpha: float = 0.10
    # wave online-update gate (plan sec. 4.6)
    replay_size: int = 512
    finetune_epochs: int = 3
    holdout_size: int = 128
    gate_epsilon: float = 0.02
    gate_skill_objective: float = 0.10
    gate_skill_halt: float = 0.0
    # stopping
    min_waves_before_stop: int = 6
    no_improve_waves: int = 3
    # dry-run lighter compute (StubEvaluator acceptance path; live keeps the above)
    dry_run_replay_size: int = 64
    dry_run_finetune_epochs: int = 1


@dataclass
class ModelConfig:
    """``[model]`` — backend selection + checkpoint location (plan sec. 4.5)."""

    backend: str = "posval_cnn"              # posval_cnn | sklearn_fallback
    model_dir: str = "data/models/20260716_195130"
    device: str = "cpu"
    library_id: str = "ga80"
    store_dir: str = "data/store"
    #: Conditioning / feature schema the encoder builds and every retrain trains
    #: under (``v2`` | ``v3`` | ``v4``; validated against
    #: :data:`lpopt.model.featurize.CHANNELS_BY_SCHEMA`).  Threaded into the
    #: curriculum retrain as ``--cond-schema``.  Default ``v3`` (do NOT flip until
    #: a v4 population is harvested and a v4 ensemble is trained, per plan 12.4).
    cond_schema: str = "v3"
    #: --- hires map-path structure (A/B arm A6, promoted 2026-07-25) -----------
    #: These MUST move together with ``cond_schema``.  A v6 schema with the legacy
    #: linear map head is the worst of both worlds: the extra input channels are
    #: paid for but the 644-parameter 1x1 readout cannot use them, and the A6
    #: result (node_peak Delta75/SD 1.41 -> 0.70) does not reproduce.  Threaded
    #: into BOTH retrain paths by ``curriculum._v5_train_config`` /
    #: ``_v5_train_flags``, so local and remote retrains stay identical.
    #: Defaults keep a deck that never mentions them on the pre-hires path.
    map_head_mode: str = "linear"
    map_prior_residual: bool = False
    map_spectral_weight: float = 0.0
    #: Campaign inference backend selector (plan sec. 4.7) — the friendly, explicit
    #: knob.  ``"local_cpu"`` scores every candidate on this PC's CPU (the network-
    #: independent default — a campaign survives a server outage).  ``"remote_gpu"``
    #: offloads the large-pool screen/deepen bulk inference to the gpu2-6000 GPU and
    #: returns only the ranked score arrays (a few MB); it probes the server first
    #: (5 s) and — on an unreachable server OR any per-batch transport error mid-
    #: campaign — logs loudly and falls back to local CPU rather than aborting.
    #: Empty ``""`` (the default) means *unset*: the legacy ``remote_screening`` key
    #: below governs, so a deck that mentions neither behaves exactly as before.
    #: ``inference`` takes precedence over ``remote_screening`` when both are set.
    inference: str = ""
    #: Offload the lean ``user_criteria`` screen/deepen bulk inference to the
    #: gpu2-6000 GPU (plan 4.7).  ``"auto"`` tries the server (5 s probe) and
    #: falls back to local CPU if unreachable; ``true`` always attempts remote
    #: (still falling back per-batch on any failure); ``false`` (default) stays
    #: fully local.  Accepts the TOML bool ``true``/``false`` or the string
    #: ``"auto"``.  Legacy alias — prefer the ``inference`` selector above.
    remote_screening: str | bool = False
    #: Only route a prediction batch to the GPU when it has at least this many
    #: predictions — smaller batches are cheaper on local CPU than the ssh
    #: round-trip.  The batched screen prewarm (~10k) clears it by default; the
    #: per-cell deepen pools (~2k) route only when this is lowered.
    remote_screening_min: int = 5000
    #: Censor Dataset A's ``max_pin_burnup`` training labels (loss mask 0) so the
    #: pin-burnup head trains only on the fidelity-consistent Dataset P (real
    #: MAS_PPI pin-resolved) labels.  Dataset A's pin label is a MOCHA-cache
    #: surrogate (near-constant 1.08x assembly burnup, 99.9% of non-null pin
    #: rows) that outweighs P ~43:1 and teaches the head a degenerate mapping,
    #: collapsing within-cell OOS pin-burnup rank skill
    #: (data/reports/pinbu_forensics.md).  ``max_pin_burnup`` is gate-ADVISORY
    #: (``[curriculum] gate_advisory_targets``), so censoring is safe to trial.
    #: Labels only — serving/inference is unaffected.  Threaded into every full
    #: retrain (local + remote) via the trainer.  DEFAULT TRUE.
    censor_dataset_a_pin_labels: bool = True
    #: --- v5 bundle knobs, threaded into every full retrain -------------------
    #: All default OFF, so a deck that does not mention them produces exactly the
    #: training command the curriculum issues today.  Flip them (together with
    #: ``cond_schema = "v5"``) only after the pre-registered A/B picks a winner.
    #: Regress the cyclen RESIDUAL against the leading-order reactivity-balance
    #: physics prior (``lpopt.model.physics_prior``); the prior is added back at
    #: serve time so ``predict()`` still returns absolute cyclen.
    cyclen_physics_prior: bool = False
    #: Add pinball-loss q10/q50/q90 heads for f_r + cyclen alongside mean/sigma.
    quantile_heads: bool = False
    #: Promote ``max_assembly_burnup`` to a first-class regression target
    #: (surrogate column 5 stops being NaN); masked wherever the label is absent.
    promote_max_asm_bu: bool = False
    #: Promote ``f_xy`` (MASTER's FXYP, pin planar peaking) to a first-class
    #: regression target APPENDED after the legacy rows, served outside the frozen
    #: seven-column surrogate contract via ``predict_fxy``; masked wherever the
    #: label is absent.  Pre-registered in
    #: ``data/reports/fxy_head_prereg_20260829.md`` (F_xy switch P4).
    promote_fxy: bool = False
    #: Fit the per-cell cyclen + F_r affine calibrations into the new model dir at
    #: the end of every retrain, instead of by hand.  DEFAULT TRUE: a retrained
    #: champion that serves uncalibrated while being gated against a calibrated
    #: incumbent is both an unfair comparison and a silent screening-recall loss.
    auto_fit_cell_calibration: bool = True


@dataclass
class ExtractConfig:
    """``[extract]`` — Dataset A extraction sources / outputs (plan 4.2, M2).

    ``workspaces`` are the MOCHA/eqlp workspace roots; each expands to its
    ``sa_2b_cache.jsonl`` + ``sa_2b_cache.stale-*.jsonl`` cache files and its
    ``runs/`` tree (case decks + ``run_meta.json``).  The three defaults expand
    to the 11 real cache files (0_Case: main + 7 stale; eqlp_ws: main + 1 stale;
    eqlp_ws_rev02: main).  Paths are resolved relative to the campaign deck.
    """

    workspaces: list[str] = field(default_factory=lambda: [
        "../2_LP/0_Case",
        "D:/eqlp_ws",
        "D:/eqlp_ws_rev02",
    ])
    store_dir: str = "data/store"
    reports_dir: str = "data/reports"
    workers: int = 8
    # -- Dataset B (3_GA_Surrogate) sources (plan 4.2, M2-B) -------------------
    #: Root of the 3_GA_Surrogate tree, relative to the campaign deck.
    ga_root: str = "../3_GA_Surrogate"
    #: Subdir (under ``ga_root``) holding the fully-hydrated GA run snapshots:
    #: ``*/stages/ga_generations_*.jsonl`` event logs plus the ``candidates`` /
    #: ``verified`` / ``master`` worker decks used to recover patterns by digest.
    ga_runs_flow: str = "runs_flow"
    #: Manifest package roots (under ``ga_root``) to ingest when readable.  Each
    #: carries ``manifest.csv`` + ``cores/<case>/<id>/loading_shf.txt``.  All are
    #: OneDrive-dehydrated today; the extractor attempts, counts, and skips them
    #: so a post-hydration re-run ingests them (idempotent by record_id).
    ga_manifest_roots: list[str] = field(default_factory=lambda: [
        "FEASIBLE_PACKAGE",
        "ga_campaign_K1_K2",
        "ga_campaign_K5_K6",
        "ga_rl_package",
    ])


@dataclass
class FuelConfig:
    """``[fuel]`` — source/output paths for the physics fuel table (plan 4.3).

    Defaults point at the real workspace assets, relative to the campaign deck
    (``5_RL/lpopt.inp``): the ``0_APR1400`` lattice tree (holding the 260624,
    5.8_5.1 and CPHA libraries), the ga80 letter-type HGCs under FEASIBLE_PACKAGE,
    the manual anchor YAML, and the persisted parquet store.
    """

    apr1400_root: str = "../0_APR1400"
    ga80_hgc: str = "../3_GA_Surrogate/FEASIBLE_PACKAGE/hgc"
    manual_yaml: str = "config/fuel_types_manual.yaml"
    store: str = "data/store/fuel_types.parquet"


@dataclass
class StratumConfig:
    """One ``[[produce.strata]]`` DoE cell (plan section 5.4).

    A stratum enumerates *which* cases (``pairs`` / ``pair_bin``, ``feed``), how
    fresh types are distributed (``center_batch`` policy, ``split_w1`` range),
    which generators mix at what weight (``generators``), how many converged
    chains are wanted (``n_target``), and the fill priority (``priority``; higher
    is served first).
    """

    name: str
    library: str = "ga80"
    #: Per-stratum ``campaign`` override for the stored rows' ``campaign`` column
    #: (``None`` -> inherit ``[produce] campaign``).  The curriculum tags each
    #: cell's produced rows with ``campaign == <cell id>`` (see
    #: ``curriculum._converged_count`` filtering on the ``campaign`` column); a
    #: multi-PC produce kit reproduces that convention by setting one stratum per
    #: cell with ``campaign`` equal to the cell id, so its rows merge back under
    #: the same per-cell key the curriculum reads.
    campaign: str | None = None
    #: Explicit case pairs (e.g. ``["K1_K2", "J1_N1"]``).
    pairs: list[str] = field(default_factory=list)
    #: Optional named pair bin (documentation only unless a resolver maps it).
    pair_bin: str | None = None
    feed: int = 121
    #: ``"auto"`` -> the pair's first fresh type; otherwise a literal batch id.
    center_batch: str = "auto"
    #: ``[lo, hi]`` (or a single value) fraction of feed weight on the pair's
    #: first type; sampled per chain to relabel fresh units.
    split_w1: list[float] = field(default_factory=lambda: [0.5])
    #: ``{generator_id: weight}`` mix over ``random`` / ``heuristic`` / ``elite_perturb``.
    generators: dict[str, float] = field(default_factory=lambda: {"random": 1.0})
    n_target: int = 24
    priority: int = 100
    allow_single_cycle_discharge: bool = False
    max_shuffle_depth: int = 2
    #: Per-stratum override of ``[produce] elite_objective`` (``None`` -> inherit).
    #: One of ``"cyclen"`` / ``"flat"`` / ``"flat_feasible"`` — see
    #: :attr:`ProduceConfig.elite_objective`.
    elite_objective: str | None = None
    notes: str = ""


@dataclass
class ProduceConfig:
    """``[produce]`` — learning-data production campaign (plan section 5.4).

    Core policy: data PRODUCTION drives the CPU as hard as it safely can, so
    ``use_all_cores`` defaults **True** — the :class:`WaveVerifier` pins one
    MASTER chain to every logical core (P-cores first, then E-cores) minus
    ``host_reserve`` core(s) held back for the Python host.  ``workers = 0`` means
    *auto* (fill that pool); a positive ``workers`` caps it.  Set
    ``use_all_cores = false`` for the legacy 8-P-core behaviour.
    """

    campaign: str = "produce"
    ledger: str = "data/produce/ledger.jsonl"
    store_dir: str = "data/store"
    workers: int = 0                         # 0 = auto (fill the core pool)
    #: fill one MASTER chain per logical core (P first, then E); the directive's
    #: "CPU 100%" default for data production.  False -> legacy 8 P-cores only.
    use_all_cores: bool = True
    #: logical cores held back for the host process when ``use_all_cores`` is on.
    host_reserve: int = 1
    chain_timeout: int = 3600
    max_cycles: int = 14
    #: Successive equilibrium cycles whose five FOMs must ALL stay within tolerance
    #: back-to-back before the chain is declared converged (vendor default 2 = two
    #: consecutive matches).  The user frames physical acceptance as ~5-cycle FOM
    #: consistency; note that raising this trades cost for confidence at roughly
    #: **+1 chained MASTER cycle per +1 consecutive** (a converged chain runs
    #: ``n_settle + consecutive`` cycles), so it is left at the vendor default and
    #: only raised deliberately.  Governs the produce / curriculum equilibrium
    #: window; ``[master] consecutive`` governs OPTIMIZE / user_criteria.
    consecutive: int = 2
    resume: bool = True
    purge_case_dirs: bool = True
    #: Delete each intermediate (pre-equilibrium) cycle's MASTER work products
    #: (MAS_RST / MAS_OUT / MAS_INP copies, staged libs) the instant the next cycle
    #: no longer needs them — keeping only the final equilibrium cycle (its MAS_SUM
    #: / final restart when kept) and a small error-diagnosis tail for failed
    #: chains (USER DIRECTIVE).  Honoured by the produce driver, curriculum produce
    #: / blind probes / mini campaigns, user_criteria verification, and bootstrap.
    purge_intermediate: bool = True
    #: Level-4 neutral warm restart path (``CaseAssetResolver`` fallback).
    neutral_restart: str | None = None
    #: Promotion cache root: converged chains' last MAS_RST.* land here for
    #: self-improving reuse (plan 5.2).
    promoted_root: str = "data/produce/promoted"
    #: Readable template-deck fallbacks (globs); the default points at the fully
    #: hydrated runs_flow GA candidate decks (plan environment reality).
    template_fallbacks: list[str] = field(default_factory=lambda: [
        "../3_GA_Surrogate/runs_flow/*/cases/*/*/candidates/*/rank_*_*/MAS_INP_cy*.inp",
    ])
    #: Cache root for SYNTHESIZED reload templates (``CaseAssetResolver`` final
    #: tier).  When no packaged/fallback deck resolves for a pair — the cross-family
    #: case (e.g. ``J2_L3``) that ships no deck — a reload template is synthesized
    #: via ``coredeck`` for the library's full roster and cached here as
    #: ``<pair>/MAS_INP_cy12.inp``, reused across campaigns (plan 5.2 / 12.1).
    #: Project-relative (NOT run-scoped) so the cache persists between runs.
    synth_decks_root: str = "data/design/synth_decks"
    #: ``rule_biased`` GENERATOR (engineering-rule R-03 bias) — OFF unless a
    #: stratum names ``rule_biased`` in its ``generators`` mix, so the default
    #: campaign is unchanged.  It draws ``random`` genomes and REJECTS any whose
    #: fresh-fresh face-adjacency count (``rule_metrics.rm_fresh_face_adjacency``)
    #: lands in the worst ``100 - rule_bias_percentile`` percent of the metric's
    #: own random-draw distribution for that (pair, feed).  The threshold is
    #: calibrated per (pair, feed) from ``rule_bias_calib`` random draws so the
    #: rejection is a DECILE of what the sampler actually produces, not a
    #: hand-picked absolute count.  It is a BIAS, not a filter: after
    #: ``rule_bias_tries`` rejected draws the last candidate is accepted, so the
    #: generator can never starve and no region is truncated out of the DoE.
    rule_bias_percentile: float = 90.0
    rule_bias_calib: int = 128
    rule_bias_tries: int = 16
    #: ``rm1`` (whole core, the measured decile rule) or ``rm1i`` (inboard pairs
    #: only — the study's stronger carrier, rho +0.235 vs +0.085).
    rule_bias_metric: str = "rm1"
    #: WHICH converged store rows the ``elite_perturb`` generator draws its
    #: PARENTS from.  The elite pool is the neighbourhood that generator explores,
    #: so selecting it by the wrong quantity aims the whole exploit arm at the
    #: wrong basin.
    #:
    #: * ``"cyclen"`` (DEFAULT, legacy) — top-32 converged rows of the pair by
    #:   ``cyclen`` DESCENDING.  Correct for a cycle-length campaign.
    #: * ``"flat"`` — top-32 converged rows of the pair that CARRY a ``node_peak``
    #:   label, ordered ``node_peak`` ASCENDING (flattest first) with ``map_cov``
    #:   ascending as the tie-break — the same primary/secondary pair the
    #:   ``flat_power`` objective ranks by.  Same-feed rows are preferred.
    #: * ``"flat_feasible"`` — ``flat`` restricted to rows that also pass the
    #:   ``flat_power`` constraint gates (CBC / F_q / |AO|) from ``[acquisition]``.
    #:
    #: MEASURED (2026-07-31): with the legacy ``cyclen`` rule the E1_E2 elite set
    #: was 100% feed-133/141 high-cyclen rows, and EVERY flat_power campaign
    #: winner (node_peak ~1.23-1.28 at cyclen ~632, feed 121) fell below the
    #: cyclen cut — so ``elite_perturb`` never once perturbed the flattest cores.
    #: Set ``"flat"`` / ``"flat_feasible"`` for a flatness (frontier) campaign.
    elite_objective: str = "cyclen"
    strata: list[StratumConfig] = field(default_factory=list)


@dataclass
class DesignConfig:
    """``[design]`` — parametric fuel-design production chain (plan section 12).

    Locates the DeCART2D / MASTER executables and the assembled ``paramA``
    package.  ``paramA_root`` (when set and present) is picked up by the fuel
    table so the new types ingest with full physics; ``package_root`` is the
    FEASIBLE_PACKAGE-layout dir the verify harness points at.
    """

    decart_exe: str = r"D:\DeCART_MASTER\BIN\decart2d1.1m5omp.exe"
    master_exe: str | None = None            # None -> fall back to [master].executable
    apr1400_root: str = "../0_APR1400"
    store_dir: str = "data/design"
    #: Assembled paramA package (lib/ bases/ cores/ hgc/ designs.json).  Also the
    #: fuel_types ``paramA_root`` ingest source.
    paramA_root: str | None = None
    package_root: str | None = None
    n_types: int = 96                        # LHS grid sample size
    seed: int = 0
    #: Concurrent DeCART2D runs.  2 is the PROVEN HOST_181 recipe (the queue
    #: script ran the serial exe two at a time); the queue was retired in favour
    #: of ``lattice.run_batch``, so this value is now the only place that recipe
    #: lives (assembly on-demand task #10 (3)).
    max_parallel: int = 2                    # concurrent DeCART2D runs
    #: Per-case DeCART wall-clock cap [s].  2.33x the HOST_199 serial measurement
    #: of 3,084 s; the previous 5400 was only 1.75x and is too thin for an
    #: authored 20-Gd lattice (task #10 (2)).
    decart_timeout: int = 7200
    bootstrap_max_cycles: int = 16
    enable_pin_burnup: bool = True
    #: Cap the throwaway cy1 fresh-core cycle at this many EFPD instead of running
    #: it to natural EOC.  ``None`` = historical behaviour.  An uncapped all-fresh
    #: cy1 runs 894-981 EFPD (34-37 MWd/kgHM), so cy02 inherits a carryover batch
    #: 1.5-2.0x deeper than the equilibrium once-burned batch it stands in for.
    #: Principled value: ``2 * B1 / (241/feed + 1)`` (linear reactivity model),
    #: ``B1`` = the uncapped cy1 EFPD.  ``--cy1-cap-efpd`` overrides per run.
    cy1_cap_efpd: float | None = None
    # optional explicit TotalBatcher tool paths (None -> resolve from apr1400_root)
    mas_ref: str | None = None
    prolog_exe: str | None = None
    totalbatcher_exe: str | None = None


@dataclass
class CurriculumConfig:
    """``[curriculum]`` — cell-sequential curriculum driver (plan section 12.2/12.3).

    The curriculum walks a ``(e_core band x feed)`` cell grid outward from a
    support anchor, and at every cell runs a crash-safe per-cell state machine
    (``ensure_types -> blind_probe -> produce_cell -> retrain -> validate_gate``)
    so the transfer methodology is *measured before* and *re-validated after*
    each increment of learning.  ``cell_order`` overrides the deterministic
    expanding-ring order with an explicit list of cell ids (``<lo>-<hi>_f<feed>``,
    e.g. ``5.25-5.5_f117``).  ``cell_pairs`` overrides the auto band->pairs
    selection for named cells.
    """

    state_dir: str = "data/curriculum"
    library: str = "ga80"
    #: e_core bands (each ``[lo, hi]``); default = the plan section-12.2 grid.
    e_core_bands: list[list[float]] = field(default_factory=lambda: [
        [5.0, 5.25], [5.25, 5.5], [5.5, 5.75],
        [5.75, 6.0], [6.0, 6.25], [6.25, 6.5],
    ])
    feeds: list[int] = field(default_factory=lambda: [101, 109, 117, 125, 133, 141])
    #: support anchor: the ``[lo, hi]`` band and feed the ring order expands from.
    anchor_band: list[float] = field(default_factory=lambda: [5.25, 5.5])
    anchor_feed: int = 117
    #: explicit cell-id order override (empty -> deterministic expanding rings).
    cell_order: list[str] = field(default_factory=list)
    # -- per-cell knobs -------------------------------------------------------
    probe_size: int = 16
    n_target: int = 150
    #: minimum usable (full-physics preferred) library types a band must have
    #: before ``ensure_types`` accepts it without on-demand design generation.
    min_band_types: int = 4
    max_pairs: int = 4                        # auto-selected in-band pairs / cell
    split_w1: list[float] = field(default_factory=lambda: [0.5])
    generators: dict[str, float] = field(default_factory=lambda: {
        "random": 0.4, "heuristic": 0.4, "elite_perturb": 0.2,
    })
    #: global explicit in-band pair override (applies to every auto cell).
    pairs: list[str] = field(default_factory=list)
    #: per-cell pair override: ``{cell_id: [pairs]}``.
    cell_pairs: dict[str, list[str]] = field(default_factory=dict)
    # -- retrain --------------------------------------------------------------
    retrain_mode: str = "remote_full"        # remote_full | local_finetune | local_full
    retrain_ensemble: int = 5
    retrain_split: str = "S1"
    replay_size: int = 512
    finetune_epochs: int = 3
    remote_poll_s: int = 60
    #: sampling-weight cap OVERRIDE applied to curriculum-cell rows only
    #: (``dataset=='P'`` with a ``campaign`` that is a known curriculum cell);
    #: the legacy A/B/P0 corpus keeps the global ``[train] cell_weight_cap`` (8.0).
    #: Raising it (default 16.0) lets previously-learned cells fully un-cap their
    #: inverse-sqrt weight (cells 1-3 uncapped ~= 12.9) so their ranking holds
    #: against a new cell's gradient pressure (negative-transfer mitigation).
    #: Threaded to the trainer via the split manifest, so no fragile CLI list.
    cell_weight_cap: float = 16.0
    # -- validation gates -----------------------------------------------------
    #: max allowed per-target within-case Spearman drop on any previous cell
    #: (the honest no-regression gate's epsilon).  Calibrated from the gate's own
    #: null distribution: at n=30 per-cell holdouts and a max-of-6 gate statistic,
    #: eps=0.05 has ~53% family-wise false-reject rate under an EQUIVALENT
    #: candidate; a 5% family-wise rate corresponds to eps~0.10 (per-check ~0.07).
    #: See data/reports/gate_noise_analysis.md.
    gate_noreg_epsilon: float = 0.10
    #: ACTIVATION SWITCH for the no-regression gate's F_r axis (user decision
    #: 2026-07-26).  ``f_r`` is always SCORED and printed; this knob decides
    #: whether its drop may VETO promotion.
    #:
    #: Default false because the veto would today punish a DATA limitation, not a
    #: model regression: measured 2026-07-26, ZERO of the 36 done cells' val
    #: holdouts (1,592 rows) carry a single ``f_r < 1.55`` label — the lowest val
    #: F_r anywhere is 1.5974 — so the gate scores bulk F_r rank two-tenths above
    #: the licensing limit, not boundary skill; and in the band where it WOULD
    #: matter the transpose-pair label ceiling is rho_max = 0.839
    #: (data/reports/transpose_noise_measured_20260725.md §2.2), i.e. below the
    #: skill a perfect physics model could show.  Core F_r is set by the hottest
    #: ASSEMBLY, so the axis only acquires learnable signal once FA-optimized
    #: assemblies are loaded — the flatness-first program's sequencing is
    #: flattening rules now -> FA optimization -> then review this switch.
    #:
    #: Flip to true ONLY when all three of
    #: :data:`..curriculum.FR_GUARD_ACTIVATION_CRITERIA` hold; the gate JSON's
    #: ``no_regression.fr_guard`` block reports the measured state of criterion (b)
    #: on every run so the decision is made against data, not memory.
    gate_noreg_fr_guard_enabled: bool = False
    #: min new-cell post-train mean within-case Spearman to pass the cell.
    gate_new_cell_min_spearman: float = 0.0
    #: PROBE targets whose per-target new-cell skill is REPORTED but does NOT
    #: drive the gate's mean-Spearman pass/fail (advisory only).  ``max_pin_burnup``
    #: is advisory by forensic verdict (data/reports/pinbu_forensics.md): it is the
    #: only probe target that fails to generalize within-cell out-of-sample
    #: (held-out Spearman ~0 across cells vs 0.65-0.95 for the other five), and at
    #: the ~11-row probe holdout its Spearman is noise-dominated (bootstrap 95% CI
    #: spans zero) — so the reported "-0.8"/"-0.37" are small-sample draws, not a
    #: real anti-correlation.  Including it injects noise into the aggregate (the
    #: pass threshold is 0.0, so a spurious negative can false-fail an otherwise
    #: healthy cell).  Its acquisition limit is already report-only (``pin_bu_limit``
    #: default ``None``); this keeps the gate consistent with that stance.
    gate_advisory_targets: list[str] = field(default_factory=lambda: ["max_pin_burnup"])
    gate_mini_budget: int = 16
    # -- legacy-corpus tail no-regression guard (plan 12.3; forensic 20260719) --
    #: The honest per-cell gate scores only ga80 curriculum cells and the global
    #: val zMAE-cyclen is tail-insensitive, so a collapse concentrated in the
    #: HIGH-cyclen Dataset-A tail (the 700-720 EFPD band, entirely the 5.8_5.1
    #: library) escaped every gate.  This guard scores BOTH champions on a fixed,
    #: stable-hash sample of S1-val Dataset-A rows in the bands below and fails when
    #: the candidate's cyclen MAE degrades by more than ``gate_tail_epsilon`` EFPD.
    #: Rows are featurized from their OWN ``library_id`` provenance
    #: (:meth:`PosValCnnBackend.predict_rows_raw`), so the score is train/serve
    #: parity-correct (not the provenance-less serve path that manifests the bug).
    gate_tail_enabled: bool = True
    #: (lo, hi) EFPD cyclen bands scored by the tail guard.
    gate_tail_bands: list[list[float]] = field(
        default_factory=lambda: [[660.0, 680.0], [680.0, 700.0], [700.0, 720.0]])
    #: per-band stable-hash sample size (n>=150 keeps the band-MAE noise < ~0.3 EFPD).
    gate_tail_sample: int = 150
    #: max allowed per-band cyclen-MAE increase (new - old) [EFPD].  Calibrated at
    #: ~2 EFPD: the champion lineage's per-band MAE is 0.7-0.9 EFPD and stable to
    #: <0.3 across retrains, so 2.0 is ~5 sigma of retrain noise while still catching
    #: the >=8 EFPD (v3) / >=37 EFPD (v4) tail collapses the forensic isolated.
    gate_tail_epsilon: float = 2.0
    #: feed the tail rows are restricted to (the fixed-feed corpus anchor).
    gate_tail_feed: int = 121
    #: mini user_criteria targets; ``None`` -> derive from the cell's live physics.
    gate_cyclen_target: float | None = None
    gate_discharge_target: float | None = None
    # ``gate_min_f_r`` was REMOVED 2026-07-26 — see :data:`RETIRED_KEYS`.  It was
    # the mini campaign's SELECTION target until the campaign moved to the
    # flat_power objective; after that move nothing read it, so it sat here
    # looking like a settable threshold while controlling nothing.  The D2
    # licensing constant it carried lives on as
    # ``curriculum.FR_GUARD_LICENSING_LIMIT`` — a module constant, which is what a
    # licensing limit is: not a knob an operator may retune.
    #: run the budget-``gate_mini_budget`` live mini campaign in validate_gate.
    gate_mini_campaign: bool = True
    # -- on-demand design generation (bands lacking full-physics types) -------
    allow_design: bool = True
    design_n_types: int = 12
    # -- per-band library resolution (plan 12.2: ga80 <= ~5.5 w/o, paramA above) -
    #: Bands whose LOWER edge (``lo``) is >= this threshold [w/o] resolve to the
    #: paramA parametric library instead of ``library`` (ga80) — the ga80 letter
    #: roster has no full-physics types above ~5.5 w/o core-average, so those
    #: bands are served by the on-demand paramA design chain.  The resolved
    #: library drives ``ensure_types`` (band-type gate + pair selection) and the
    #: per-cell produce stratum.
    paramA_band_lo: float = 5.75
    #: Library id the ``lo >= paramA_band_lo`` bands resolve to.
    paramA_library: str = "paramA"
    #: Optional explicit ``{band_label: library_id}`` override map (e.g.
    #: ``{"5.75-6" = "paramA"}``); a matching entry takes precedence over the
    #: ``paramA_band_lo`` threshold rule.
    band_libraries: dict[str, str] = field(default_factory=dict)


@dataclass
class CriteriaConfig:
    """``[criteria]`` — FREE-SEARCH ``user_criteria`` campaign (plan sec. 12.5).

    The outer decision variable is the fuel PAIR (+ its batch split); the search
    picks WHICH two fresh types to feed and the LP that arranges them, subject to
    a target core-average enrichment band.  Objective hierarchy (per
    :func:`lpopt.search.acquisition.score_user_criteria`): cyclen within tolerance
    of target -> discharge-burnup criterion band -> MINIMIZE F_r on its UCB.  All
    seven constraint axes are user-settable; ``None`` means report-only (not
    gated).  ``mtc_limit`` / ``sdm_limit`` are NOT model-predicted — they are
    carried here only for the post-verification stage (``lpopt.search.sdm_mtc``).

    Reachability (the 2_LP feasible-pair lesson): a pair enters the universe only
    when its achievable mass-weighted ``e_core`` interval over
    ``split in split_range`` overlaps ``e_core_target +/- e_core_tol``.  Unreachable
    pairs are excluded from the denominator (never wasted budget).
    """

    # -- e_core band (pair/split reachability + per-candidate screen) --------- #
    e_core_target: float = 5.2
    e_core_tol: float = 0.05
    #: ``[lo, hi]`` batch-split fraction searched when testing pair reachability.
    split_range: list[float] = field(default_factory=lambda: [0.2, 0.8])
    #: allow A==B mono pairs in the universe (when the deck/resolver supports one
    #: fresh type); a mono pair is included iff its single enrichment is in band.
    allow_mono: bool = True

    # -- target bands (both model-predicted) --------------------------------- #
    cyclen_target: float = 625.0
    cyclen_tol: float = 2.0
    discharge_target: float | None = None
    discharge_tol: float = 0.5

    # -- the 7 constraint axes (None -> report-only, not gated) -------------- #
    f_r_limit: float | None = 1.55          # also the minimization objective
    f_q_limit: float | None = 2.41
    cbc_limit: float | None = 1550.0
    asi_abs_limit: float | None = 0.30      # |ASI| == |AO| numerically
    pin_bu_limit: float | None = None       # default report-only (plan 12.5)
    mtc_limit: float | None = None          # post-verification only
    sdm_limit: float | None = None          # post-verification only
    #: UCB risk shift kappa on the F_r objective, the target distances, and gates.
    risk_z: float = 0.25

    # -- search strategy (plan sec. 12.5 addendum: predict-then-verify) ------ #
    #: ``"lean"`` (default) — one-shot surrogate screen of the full pair universe
    #: + deepened pools on the top cells, then ONE batched MASTER verification wave
    #: of the global top-K predicted candidates (answers in minutes, no per-wave
    #: fine-tuning).  ``"active"`` — the outer racing/waves allocation below.
    search_mode: str = "lean"

    # -- lean (predict-then-verify) knobs ------------------------------------ #
    lean_deep_cells: int = 16               # top screen cells given a deepened pool
    lean_pool_per_cell: int = 2000          # deepen pool size / cell (surrogate-only)
    lean_top_k: int = 12                    # verified predicted candidates (one wave)
    lean_per_pair_cap: int = 3              # max verified entries per pair-cell
    lean_hamming_min: int = 4               # pairwise Hamming floor among the top-K
    #: STORE-verified feasible LPs injected into each cell's screen/deepen pool as
    #: elite parents (converged, in e_core band) so a known-good basin is deepened
    #: and its small-move mutation children become NEW verifiable candidates near
    #: the known optimum.  0 disables the injection (legacy behaviour).
    lean_store_elites_per_cell: int = 8
    #: if NO verified candidate meets the criteria bands, run ONE more top-K round
    #: informed by the residuals (default OFF — keep the lean promise truly lean).
    lean_second_round: bool = False

    # -- outer allocation over the pair universe (plan sec. 6.2 racing) ------ #
    outer_max_cells: int = 8                # cells activated after wave-0 screen
    outer_screen_budget: int = 24           # verify slots spent during racing
    outer_target_cells: int = 3             # racing target survivor count
    outer_exploit_floor: int = 5            # min exploit slots on the best cell
    outer_verify_per_wave: int = 2          # verify slots / cell / racing wave
    outer_race_z: float = 1.0               # racing UCB/LCB confidence multiplier
    outer_softmax_temp: float = 1.0         # exploit softmax temperature
    screen_pool_per_cell: int = 12          # wave-0 surrogate screen pool / cell

    # -- report / post-verification ------------------------------------------ #
    post_verify_topk: int = 3               # SDM/MTC post-verify final top-K (0=off)

    # -- post-hoc discharge energy-balance estimate (APR1400 defaults) ------- #
    power_mw: float = 3983.0
    hm_mtu: float = 104.8

    def validate(self) -> None:
        if self.search_mode not in ("lean", "active"):
            raise ConfigError(
                f"[criteria] search_mode {self.search_mode!r} invalid; expected "
                "'lean' or 'active'"
            )
        if self.e_core_tol < 0:
            raise ConfigError(f"[criteria] e_core_tol {self.e_core_tol} must be >= 0")
        sr = list(self.split_range)
        if len(sr) != 2 or not (0.0 <= sr[0] <= sr[1] <= 1.0):
            raise ConfigError(
                f"[criteria] split_range {self.split_range} must be [lo, hi] "
                "with 0 <= lo <= hi <= 1"
            )
        if self.cyclen_tol < 0 or self.discharge_tol < 0:
            raise ConfigError("[criteria] cyclen_tol / discharge_tol must be >= 0")
        if self.outer_max_cells < 1:
            raise ConfigError("[criteria] outer_max_cells must be >= 1")
        if self.outer_target_cells < 1:
            raise ConfigError("[criteria] outer_target_cells must be >= 1")
        if self.lean_top_k < 1:
            raise ConfigError("[criteria] lean_top_k must be >= 1")
        if self.lean_per_pair_cap < 1:
            raise ConfigError("[criteria] lean_per_pair_cap must be >= 1")


@dataclass
class ConstraintsConfig:
    """``[constraints]`` — USER-SET licensing limits for the post-verification axes.

    Decision D9 (2026-07-25): *"F_r 제외 feasible 노심에 대해서 (평탄도 높음) SDM,
    MTC 검증 실시"* — the flatness objective is free to flatten because F_r left the
    objective, but flattening monotonically degrades control-rod worth and raises
    leakage, and **no search axis checks either**.  SDM / MTC therefore become a
    MANDATORY PRE-DELIVERY GATE, run after the campaign picks its top-K flat
    feasible (= feasible EXCLUDING F_r) candidates.

    Division of labour with ``[sdm_mtc]``: this section is the **user knob surface**
    — *whether* an axis runs and *what limit* it is judged against.  ``[sdm_mtc]``
    keeps the **branch mechanics** (temperature step, rod banks, timeout, sidecar
    path), which are physics-harness settings, not user preferences.

    ``enable`` and ``limit`` are INDEPENDENT (the plan's requirement):

    * ``*_enable = false`` (**default**) — the axis is not run at all.  Running it
      costs real MASTER calls, so it is never implicit: a campaign's cost may not
      change because a default flipped.
    * ``*_enable = true`` + limit unset (**default when enabled**) — REPORT-ONLY.
      The branch runs, the number is measured, recorded and printed, and NOTHING
      is ever marked a violator.  This is the honest state while the user's own
      thresholds are unknown.
    * ``*_enable = true`` + limit set — GATED.  A candidate outside the limit is
      marked a violator in ``delivery.json`` / the verdict table.

    **OPEN USER QUESTION — the limits below are deliberately unset.**  We do not
    know this project's licensing thresholds.  The physically standard APR1400
    values (DCD Table 4.3, and what MOCHA uses as its constants) are offered as a
    SUGGESTION only, and must be confirmed before any gate is switched on:

    * ``mtc_max_pcm_per_c = 9.0`` — most-POSITIVE allowed MTC [pcm/°C]
      (BOC/HZP end of the window; a positive MTC above this fails);
    * ``mtc_min_pcm_per_c = -54.0`` — most-negative allowed MTC [pcm/°C];
    * ``sdm_required_pcm = 10870.0`` — minimum required shutdown margin [pcm]
      (= CEA allowance 10180 + net-worth uncertainty 690).
    """

    #: Run the MTC branch on the selected candidates (~1 extra MASTER call each).
    mtc_enable: bool = False
    #: Most-POSITIVE allowed MTC [pcm/°C].  ``None`` -> report-only (never gated).
    mtc_max_pcm_per_c: float | None = None
    #: Most-negative allowed MTC [pcm/°C].  ``None`` -> that edge is not gated.
    mtc_min_pcm_per_c: float | None = None

    #: Run the SDM branch (needs ``[sdm_mtc]`` rod-model assets; see module docs).
    sdm_enable: bool = False
    #: Minimum required shutdown margin [pcm].  ``None`` -> report-only.
    sdm_required_pcm: float | None = None

    #: How many of the campaign's delivery-ranked flat feasible candidates go to
    #: post-verification (0 = off).  Each costs ~1 MASTER call per enabled axis.
    post_verify_top_k: int = 3

    def mtc_gated(self) -> bool:
        """True iff an MTC verdict may be a PASS/FAIL rather than report-only."""
        return bool(self.mtc_enable) and (
            self.mtc_max_pcm_per_c is not None or self.mtc_min_pcm_per_c is not None
        )

    def sdm_gated(self) -> bool:
        """True iff an SDM verdict may be a PASS/FAIL rather than report-only."""
        return bool(self.sdm_enable) and self.sdm_required_pcm is not None

    def any_enabled(self) -> bool:
        return bool(self.mtc_enable or self.sdm_enable)

    def validate(self) -> None:
        if self.post_verify_top_k < 0:
            raise ConfigError(
                f"[constraints] post_verify_top_k {self.post_verify_top_k} must be >= 0"
            )
        lo, hi = self.mtc_min_pcm_per_c, self.mtc_max_pcm_per_c
        if lo is not None and hi is not None and float(lo) > float(hi):
            raise ConfigError(
                f"[constraints] mtc_min_pcm_per_c {lo} must be <= mtc_max_pcm_per_c {hi}"
            )
        if self.sdm_required_pcm is not None and float(self.sdm_required_pcm) < 0:
            raise ConfigError("[constraints] sdm_required_pcm must be >= 0 pcm")


@dataclass
class SdmMtcConfig:
    """``[sdm_mtc]`` — SDM/MTC post-verification limits + branch knobs (plan 12.5).

    Additive, all-optional section for the ``lpopt sdm-mtc`` post-verification
    stage (``lpopt.search.sdm_mtc``).  Units mirror the MOCHA/APR1400 DCD Table
    4.3 licensing convention: MTC in **pcm/°C** (``mtc_max`` = most-positive
    allowed), SDM in **pcm** (``sdm_required`` = minimum required net worth).
    """

    top_k: int = 5
    mtc_min_pcm_per_c: float = -54.0
    mtc_max_pcm_per_c: float = 9.0
    sdm_required_pcm: float = 10870.0
    cea_allowance_pcm: float = 10180.0
    net_worth_uncertainty_pcm: float = 690.0
    mtc_delta_c: float = 5.0
    mtc_output_units: str = "pcm_per_c"          # pcm_per_c | drho_per_c_1e-4
    # Scram / stuck-search scope MUST include the shutdown banks A and B.  The
    # regulating banks alone carry only ~26 % of total CEA worth (A+B are 12.32
    # of 16.70 %drho at EOC, DCD Table 4.3-6), so an R-only default cannot reach
    # the 10,870 pcm requirement for ANY pattern — every candidate would FAIL
    # for a reason that is a config defect, not physics.  This mirrors the
    # validated MOCHA scope (2_LP/.claude/skills/master-sdm-mtc/references/
    # dcd-limits.md: "Default scram and stuck search includes R1-R5, B, and A").
    # PSCEA ``P`` stays parsed-but-excluded (DCD "Total without PSCEA").
    scram_banks: list[str] = field(
        default_factory=lambda: ["R1", "R2", "R3", "R4", "R5", "B", "A"]
    )
    stuck_candidate_banks: list[str] = field(
        default_factory=lambda: ["R1", "R2", "R3", "R4", "R5", "B", "A"]
    )
    branch_timeout_s: float = 300.0
    sidecar_path: str = "data/sdm_mtc/results.jsonl"


@dataclass
class DebugPanelConfig:
    """``[debug_panel]`` — MASTER-verified scoring panel (``lpopt debug-panel``).

    The panel is the standing answer to the 2026-07-29 directive that every model
    build be debugged against MASTER rather than against its own validation loss:
    a fixed set of MASTER-verified store rows, scored in NEUTRONICS UNITS with a
    per-target tolerance an engineer can defend.

    ``campaigns`` are fnmatch globs over the store's ``campaign`` column — panel
    membership is data, not a hand-maintained id list, so growing the panel is a
    production run and nothing else.  ``tolerances`` overrides
    :data:`lpopt.tools.debug_panel.DEFAULT_TOLERANCES` per key (an absent key keeps
    its default); the defaults are cyclen 3.0 EFPD, cbc_max 20 ppm, f_r 0.05,
    f_q 0.08, ao_abs 0.010, node_peak 0.05, map_cov 0.02.

    Report-only: nothing in this section can fail a build.  ``lpopt debug-panel
    score`` always exits 0 — it mirrors the warn-don't-block gate decision of
    2026-07-26, because a tolerance that halts a pipeline gets widened until it
    stops halting the pipeline.
    """

    campaigns: list[str] = field(
        default_factory=lambda: ["debug_panel*", "democase*"])
    tolerances: dict[str, float] = field(default_factory=dict)


# Ordered map of section name -> dataclass.  Also defines the set of allowed
# top-level sections for unknown-section detection.
_SECTIONS: dict[str, type] = {
    "flow": FlowConfig,
    "remote": RemoteConfig,
    "master": MasterConfig,
    "verify": VerifyConfig,
    "data": DataConfig,
    "case": CaseConfig,
    "fuel": FuelConfig,
    "extract": ExtractConfig,
    "produce": ProduceConfig,
    "search": SearchConfig,
    "acquisition": AcquisitionConfig,
    "model": ModelConfig,
    "design": DesignConfig,
    "curriculum": CurriculumConfig,
    "criteria": CriteriaConfig,
    "constraints": ConstraintsConfig,
    "sdm_mtc": SdmMtcConfig,
    "debug_panel": DebugPanelConfig,
}


@dataclass
class LpoptConfig:
    """Fully parsed campaign deck."""

    flow: FlowConfig
    remote: RemoteConfig
    master: MasterConfig
    verify: VerifyConfig
    data: DataConfig
    case: CaseConfig
    fuel: FuelConfig
    extract: ExtractConfig
    produce: ProduceConfig
    # M4 sections default so existing direct constructors keep working; ``load_config``
    # always supplies them.
    search: SearchConfig = field(default_factory=SearchConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    criteria: CriteriaConfig = field(default_factory=CriteriaConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    sdm_mtc: SdmMtcConfig = field(default_factory=SdmMtcConfig)
    debug_panel: DebugPanelConfig = field(default_factory=DebugPanelConfig)
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable plain dict of the six sections (no source_path)."""
        return {name: dataclasses.asdict(getattr(self, name)) for name in _SECTIONS}


#: Keys deliberately REMOVED, mapped to what replaced them.
#:
#: A dead knob has three possible fates and only one of them is honest: leave it
#: live-looking and inert (the worst — an operator sets it and nothing happens),
#: keep it with a deprecation note (right when a deck in the wild still sets it),
#: or remove it.  ``gate_min_f_r`` was removed: a repo-wide search over 2,021
#: ``*.inp`` decks on 2026-07-26 found ZERO that set it, so no archived deck
#: breaks.  A deck that does set it is wrong, but it is not a TYPO, so it gets
#: this message instead of the generic "typo?" list — otherwise the reader hunts
#: for a misspelling that was never there.
RETIRED_KEYS: dict[tuple[str, str], str] = {
    ("curriculum", "gate_min_f_r"): (
        "RETIRED 2026-07-26: the validate_gate mini campaign now selects on the "
        "flat_power objective (node_peak primary, map_cov secondary), so this "
        "threshold decided nothing.  F_r survives there as the D1 in-loop SAFETY "
        "veto (1.70) and as the reported per-row margin against the D2 licensing "
        "constant 1.55 — both are module constants "
        "(curriculum.FR_GUARD_LICENSING_LIMIT), not knobs.  Delete the line."),
}


def _build_section(cls: type, table: dict[str, Any], name: str, unknown: list[str],
                   retired: list[str] | None = None) -> Any:
    field_names = {f.name for f in fields(cls)}
    for key in table:
        if key in field_names:
            continue
        note = RETIRED_KEYS.get((name, key))
        if note is not None and retired is not None:
            retired.append(f"[{name}] {key} — {note}")
        else:
            unknown.append(f"[{name}] {key}")
    kwargs = {k: v for k, v in table.items() if k in field_names}
    return cls(**kwargs)


def _build_produce_section(table: dict[str, Any], unknown: list[str]) -> ProduceConfig:
    """Build ``[produce]`` including its ``[[produce.strata]]`` array of tables.

    ``tomllib`` parses ``[[produce.strata]]`` into ``table["strata"]`` as a list
    of plain dicts; each is validated and materialized into a
    :class:`StratumConfig` with the same unknown-key discipline as every other
    section.
    """

    field_names = {f.name for f in fields(ProduceConfig)}
    for key in table:
        if key not in field_names:
            unknown.append(f"[produce] {key}")

    raw_strata = table.get("strata", []) or []
    if not isinstance(raw_strata, list):
        unknown.append("[produce] strata must be an array of tables")
        raw_strata = []

    strat_fields = {f.name for f in fields(StratumConfig)}
    strata: list[StratumConfig] = []
    for index, entry in enumerate(raw_strata):
        if not isinstance(entry, dict):
            unknown.append(f"[[produce.strata]] #{index} must be a table")
            continue
        for key in entry:
            if key not in strat_fields:
                unknown.append(f"[[produce.strata]] #{index} {key}")
        if "name" not in entry:
            unknown.append(f"[[produce.strata]] #{index} missing required 'name'")
            continue
        kwargs = {k: v for k, v in entry.items() if k in strat_fields}
        strata.append(StratumConfig(**kwargs))

    scalar = {k: v for k, v in table.items() if k in field_names and k != "strata"}
    return ProduceConfig(strata=strata, **scalar)


def _build_search_section(table: dict[str, Any], unknown: list[str]) -> SearchConfig:
    """Build ``[search]`` including its ``[search.trust_region]`` /
    ``[search.local_search]`` sub-tables with the same unknown-key discipline."""

    field_names = {f.name for f in fields(SearchConfig)}
    for key in table:
        if key not in field_names:
            unknown.append(f"[search] {key}")

    tr_table = table.get("trust_region", {})
    if not isinstance(tr_table, dict):
        unknown.append("[search.trust_region] must be a table")
        tr_table = {}
    trust_region = _build_section(
        TrustRegionConfig, tr_table, "search.trust_region", unknown
    )

    ls_table = table.get("local_search", {})
    if not isinstance(ls_table, dict):
        unknown.append("[search.local_search] must be a table")
        ls_table = {}
    local_search = _build_section(
        LocalSearchConfig, ls_table, "search.local_search", unknown
    )

    scalar = {
        k: v
        for k, v in table.items()
        if k in field_names and k not in ("trust_region", "local_search")
    }
    return SearchConfig(
        trust_region=trust_region, local_search=local_search, **scalar
    )


def load_config(path: str | Path) -> LpoptConfig:
    """Load and validate a TOML campaign deck into an :class:`LpoptConfig`.

    Raises :class:`ConfigError` if the file is missing, malformed, or contains
    any unknown section or key (the message lists every offending item).
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"deck not found: {path}")
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    unknown: list[str] = []
    retired: list[str] = []

    # Unknown top-level sections / scalars.
    for key, value in raw.items():
        if key not in _SECTIONS:
            unknown.append(f"[{key}] (unknown top-level section)")
        elif not isinstance(value, dict):
            unknown.append(f"[{key}] must be a table, got {type(value).__name__}")

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        table = raw.get(name, {})
        if not isinstance(table, dict):
            table = {}  # already flagged above
        if name == "produce":
            sections[name] = _build_produce_section(table, unknown)
        elif name == "search":
            sections[name] = _build_search_section(table, unknown)
        else:
            sections[name] = _build_section(cls, table, name, unknown, retired)

    # A retired key is still a hard error — a deck that sets it is describing a
    # behaviour the code no longer has — but it is reported as RETIRED with its
    # replacement, not filed under "typo?".  Both groups are reported in ONE
    # message: raising on the retired list alone would hide a genuine typo until
    # the operator fixed the retired line and re-ran.
    if retired or unknown:
        parts = []
        if retired:
            parts.append("RETIRED key(s):\n  " + "\n  ".join(retired))
        if unknown:
            parts.append("unknown key(s)/section(s) (typo?):\n  "
                         + "\n  ".join(unknown))
        raise ConfigError(f"invalid deck {path}:\n" + "\n".join(parts))

    _validate_cond_schema(sections["model"].cond_schema, path)
    _validate_inference(sections["model"].inference, path)
    _validate_objective(sections["acquisition"].objective, path)
    _validate_policy_prior(sections["acquisition"].policy_prior, path)
    _validate_ood_policy(sections["acquisition"].ood_policy, path)
    _validate_conformal_alpha(sections["acquisition"].conformal_alpha, path)

    return LpoptConfig(source_path=path, **sections)


# --------------------------------------------------------------------------- #
# THE F_r guard switch — one setting, every promotion surface
#
# The user's 2026-07-26 decision is about PROMOTION/ACCEPTANCE semantics, not
# about one function: gating on F_r skill today blocks a model for a DATA
# limitation, not for a regression.  Every surface that can withhold promotion on
# an F_r drop therefore resolves its enforcement through THIS switch, so the
# FA-optimized phase re-arms them all from one place:
#
#   * ``curriculum.gate_no_regression``   — the retrain no-regression gate
#   * ``curriculum._gate_newcell``        — the new-cell skill gate (ANDs into the
#                                           same ``validate_gate`` verdict)
#   * ``model.flat_ab.judge_arm``         — offline A/B condition 5 (M5) and the
#                                           M6 extended gate it consumes
#   * ``model.ab_eval.no_regression_gate``— the offline A/B proxy gate
#
# ``curriculum._gate_mini_campaign`` is the fifth surface the same decision
# reached, but it is NOT on this switch: F_r was its SELECTION objective, and the
# flatness-first program (§1.2/§10) settles what a validate_gate campaign
# demonstrates independently of whether an F_r regression may veto a promotion.
# It keeps F_r only as the D1 in-loop safety veto, which is not a promotion gate.
#
# ``model.ab_decide.evaluate_arms`` is the fifth surface: BOTH of its point rules
# can withhold a promotion on an F_r drop (the Delta75/SD regression veto over
# ``ab_eval.PRIMARY_TARGETS``, which contains ``f_r``, and the secondary
# within-cell rho rule over ``("cyclen", "f_r")``), so both resolve here too.
#
# ``model.ab_score.score_arm`` is the sixth: the per-arm fold gate it writes is
# what ``ab_decide`` then consumes as ``passes_gate``.  It used to call
# ``ab_eval.no_regression_gate`` with no setting and no deck, so it alone stayed
# at the dataclass default while every other surface re-armed — the split-brain
# this block exists to deny.  It now resolves through :func:`fr_guard_from_deck`
# like the offline A/B, and ``ab_decide`` refuses a gate stamped with the other
# setting rather than laundering it into a pass.
#
# ``curriculum._gate_mini_campaign`` is the surface the same decision reached but
# which is NOT on this switch: F_r was its SELECTION objective, and the
# flatness-first program (§1.2/§10) settles what a validate_gate campaign
# demonstrates independently of whether an F_r regression may veto a promotion.
# It keeps F_r only as the D1 in-loop safety veto, which is not a promotion gate.
#
# The in-loop paths take the value from the loaded deck
# (``cfg.curriculum.gate_noreg_fr_guard_enabled``).  The OFFLINE A/B harness runs
# from a CLI rather than from a loaded campaign, so it resolves the SAME field
# off a deck when one is readable (:func:`fr_guard_from_deck`) and falls back to
# the field's documented DEFAULT otherwise — one dataclass default, not a second
# copy of the policy.  That is what makes the switch a single switch: flipping
# ``[curriculum] gate_noreg_fr_guard_enabled`` in ``lpopt.inp`` re-arms the
# curriculum gates AND the offline A/B, instead of re-arming only the former and
# leaving the A/B silently deferred.  ``lpopt.inp`` here means THE deck beside the
# package (:data:`FR_GUARD_DEFAULT_DECK`, absolute), not whatever file of that
# name happens to sit in the shell's working directory.
# --------------------------------------------------------------------------- #
#: Human-readable name of the switch, quoted verbatim in every gate's note.
FR_GUARD_KNOB = "[curriculum] gate_noreg_fr_guard_enabled"

#: Repo/package root — ``<root>/lpopt/config.py`` -> ``<root>``.  The campaign
#: deck lives beside the package, not inside it.
FR_GUARD_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The deck an offline (deckless) harness reads the switch off by default.
#:
#: ABSOLUTE, and deliberately so.  It used to be the bare relative name
#: ``"lpopt.inp"``, which :func:`fr_guard_from_deck` resolved against the
#: PROCESS's working directory — so the offline A/B read the real deck when it
#: was launched from the repo root and silently fell back to the deferred
#: dataclass default from anywhere else.  A policy that changes with where the
#: command was typed is not one switch, and the failure is invisible: the
#: fallback and the honest "deck says deferred" answer look identical at today's
#: setting.  Anchoring it to the package's own root makes the resolution
#: reproducible from any cwd, and when the deck genuinely is not there
#: :func:`fr_guard_from_deck` names the absolute path it looked for in
#: ``deck_error`` rather than leaving a reader to guess.
FR_GUARD_DEFAULT_DECK = str(FR_GUARD_REPO_ROOT / "lpopt.inp")


def fr_guard_enforced(fr_guarded: bool | None = None,
                      curriculum: "CurriculumConfig | None" = None) -> bool:
    """Resolve THE F_r-guard switch for any promotion surface.

    Precedence: an explicit ``fr_guarded`` (a caller that already read a deck, or
    a test pinning a branch) -> a supplied :class:`CurriculumConfig` -> the
    field's default.  Never reads global state, so a surface can be exercised at
    either setting without touching a file.
    """
    if fr_guarded is not None:
        return bool(fr_guarded)
    if curriculum is not None:
        return bool(getattr(curriculum, "gate_noreg_fr_guard_enabled", False))
    return bool(CurriculumConfig().gate_noreg_fr_guard_enabled)


def fr_guard_from_deck(deck: "str | Path | None" = FR_GUARD_DEFAULT_DECK,
                       fr_guarded: bool | None = None) -> dict[str, Any]:
    """Resolve the switch for an OFFLINE surface, and say where it came from.

    An offline harness has no loaded :class:`LpoptConfig`, which is exactly how
    the split-brain arose: the in-loop gates read the deck while the A/B read the
    dataclass default, so flipping the knob re-armed one half of the "one switch".
    This resolves the SAME field with an explicit, reported precedence:

    1. an explicit ``fr_guarded`` (a CLI override or a test pinning a branch);
    2. ``[curriculum] gate_noreg_fr_guard_enabled`` in ``deck``, when the deck
       exists and parses;
    3. the field's documented default.

    Returns the ``{"enforced", "source", "knob", "deck"}`` block a surface stamps
    on its artifact, so a reader can tell "deferred because the deck says so" from
    "deferred because nothing was read".  Never raises: an unreadable or invalid
    deck degrades to the default WITH the parse error named, because a malformed
    deck must not silently look like a policy decision.
    """
    if fr_guarded is not None:
        return {"enforced": bool(fr_guarded), "source": "explicit",
                "knob": FR_GUARD_KNOB, "deck": (str(deck) if deck else None)}
    if deck:
        p = Path(deck)
        if p.exists():
            try:
                cfg = load_config(p)
            except ConfigError as exc:
                return {"enforced": bool(CurriculumConfig().gate_noreg_fr_guard_enabled),
                        "source": "default", "knob": FR_GUARD_KNOB,
                        "deck": str(p), "deck_error": str(exc)}
            return {"enforced": bool(cfg.curriculum.gate_noreg_fr_guard_enabled),
                    "source": "deck", "knob": FR_GUARD_KNOB, "deck": str(p)}
        return {"enforced": bool(CurriculumConfig().gate_noreg_fr_guard_enabled),
                "source": "default", "knob": FR_GUARD_KNOB, "deck": str(p),
                "deck_error": f"deck not found: {p}"}
    return {"enforced": bool(CurriculumConfig().gate_noreg_fr_guard_enabled),
            "source": "default", "knob": FR_GUARD_KNOB, "deck": None}


#: Accepted ``[acquisition] objective`` values.
_VALID_OBJECTIVES = {"target_cycle", "max_cycle_min_fr", "min_fr_max_cycle",
                     "min_fuel_cost", "fr_boundary", "flat_power", "min_fxy"}


def _validate_objective(value: str, path: Path) -> None:
    if str(value).strip() not in _VALID_OBJECTIVES:
        raise ConfigError(
            f"[acquisition] objective {value!r} invalid; expected one of "
            f"{sorted(_VALID_OBJECTIVES)}  (in {path})"
        )


#: Accepted ``[acquisition] policy_prior`` values.  ``"fr"`` / ``"flat"`` /
#: ``"both"`` are v1's two heads and their mean, kept verbatim so every existing
#: deck keeps its meaning; ``"v1"`` is the version-explicit spelling of
#: ``"both"``; ``"v2"`` and ``"shadow_v2"`` select the v2 ensemble, the latter
#: for recording only.  Mapped to (family, head) by
#: :data:`lpopt.search.construct.POLICY_MODES`, which is the single source of
#: truth; ``tests/test_config.py`` asserts the two lists agree (the import is
#: deliberately NOT taken here — ``lpopt.search`` imports this module).
_VALID_POLICY_PRIORS = {"off", "fr", "flat", "both", "v1", "v2", "shadow_v2",
                        "v3", "shadow_v3"}


def _validate_policy_prior(value: str, path: Path) -> None:
    if str(value).strip().lower() not in _VALID_POLICY_PRIORS:
        raise ConfigError(
            f"[acquisition] policy_prior {value!r} invalid; expected one of "
            f"{sorted(_VALID_POLICY_PRIORS)}  (in {path})"
        )


#: Accepted ``[acquisition] ood_policy`` values (review §6.5: OOD_OK /
#: OOD_ESCALATE / OOD_REJECT).  ``"warn"`` is the shipped report-only behaviour.
_VALID_OOD_POLICIES = {"warn", "escalate", "reject"}


def _validate_ood_policy(value: str, path: Path) -> None:
    if str(value).strip().lower() not in _VALID_OOD_POLICIES:
        raise ConfigError(
            f"[acquisition] ood_policy {value!r} invalid; expected one of "
            f"{sorted(_VALID_OOD_POLICIES)}  (in {path})"
        )


def _valid_conformal_alphas() -> set[float]:
    """Miscoverage levels a conformal artifact is actually FIT at.

    Sourced from :mod:`lpopt.model.conformal` so the deck knob and the fitter can
    never drift apart; an import hiccup falls back to the module's own defaults
    rather than masking a real deck error.
    """
    try:
        from .model.conformal import DEFAULT_ALPHAS
        return {float(a) for a in DEFAULT_ALPHAS}
    except Exception:  # noqa: BLE001
        return {0.10, 0.32}


def _validate_conformal_alpha(value: float, path: Path) -> None:
    valid = _valid_conformal_alphas()
    if not any(abs(float(value) - a) <= 1.0e-9 for a in valid):
        raise ConfigError(
            f"[acquisition] conformal_alpha {value!r} invalid; the artifact is "
            f"fit only at {sorted(valid)}  (in {path})"
        )


def _valid_cond_schemas() -> set[str]:
    """Allowed ``[model] cond_schema`` values, sourced from the featurize module
    so the config stays in lock-step with the encoder's schema inventory."""
    try:
        from .model.featurize import CHANNELS_BY_SCHEMA
        return set(CHANNELS_BY_SCHEMA)
    except Exception:  # noqa: BLE001 — never let an import hiccup mask a real deck
        return {"v2", "v3", "v4"}


def _validate_cond_schema(value: str, path: Path) -> None:
    valid = _valid_cond_schemas()
    if value not in valid:
        raise ConfigError(
            f"[model] cond_schema {value!r} invalid; expected one of "
            f"{sorted(valid)}  (in {path})"
        )


#: Accepted ``[model] inference`` spellings (empty = unset -> legacy behaviour).
_VALID_INFERENCE = {"", "local_cpu", "local", "cpu", "remote_gpu", "remote", "gpu"}


def _validate_inference(value: str, path: Path) -> None:
    if str(value).strip().lower() not in _VALID_INFERENCE:
        raise ConfigError(
            f"[model] inference {value!r} invalid; expected 'local_cpu' or "
            f"'remote_gpu' (in {path})"
        )
