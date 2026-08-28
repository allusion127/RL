"""MASTER wave verification harness (``WaveVerifier``) — plan sections 4.6 / 5.4.

This is a faithful port of the snapshot ``master_rl/flow.py`` ``_ga_case_evaluator``
(L263-309): one MASTER equilibrium runner per performance-core worker, wrapped in
the vendor :class:`~lpopt.vendor.masterrl.parallel.ParallelPatternEvaluator`
scheduling model (P-cores own the physics, the host retreats to E-cores).

Differences from the GA evaluator (all additive, none touch the wiring):

* a *produce* wave mixes **heterogeneous cases** (different pair/feed/restart),
  whereas the GA evaluates one case per run.  ``evaluate_wave`` therefore builds
  a per-entry :class:`~lpopt.vendor.masterrl.dataset.CaseData` and dispatches each
  worker its own ``(case_data, pattern)`` — reusing the vendor per-worker
  evaluators and ``host_affinity`` window, but driving the fan-out itself;
* a fallback restart (level >= 1) does not match the template deck's restart
  reference, so before staging, each deck's restart reference is rewritten via
  :meth:`CaseAssetResolver.prepare_cycle1_deck` (the vendor
  ``advance_cycle_deck`` mechanism);
* the constructor takes an ``evaluator_factory`` so tests inject a
  :class:`~lpopt.search.stub.StubEvaluator` without touching MASTER.

Outcomes are a three-way taxonomy for honest budget accounting (plan 4.6):
``converged`` / ``nonconverged`` / ``error``.  :func:`outcome_to_record` maps an
outcome to a :class:`~lpopt.data.schema.CanonicalRecord` (``dataset="P"``).

For QC counting the ``error`` bucket splits further (plan 5.4 / 4.6 정직 회계):
:func:`classify_outcome` separates a **physics kill** (``non_finite_flux`` — the
NaN watchdog killing a divergent random pattern, an honest negative property of
the ``(pattern, restart)`` pair, like a non-convergence) from a genuine
**harness error** (staging / deck / resolver / timeout / genome defect).  Only
the latter reflects a harness fault; the former must never halt a stratum.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from ..data.flatness import record_flatness
from ..data.schema import (
    SYM_CLASS,
    CanonicalRecord,
    compute_record_id,
    pack_pattern,
)
from ..vendor.masterrl.dataset import CaseData
from ..vendor.masterrl.domain import FOM, CaseKey, Pattern
from ..vendor.masterrl.equilibrium import (
    EquilibriumRunError,
    EquilibriumRunner,
    deck_cycle,
)
from ..vendor.masterrl.master import MasterRunError, MasterRunner
from ..vendor.masterrl.parallel import detect_core_layout, host_affinity
from .assets import (
    LIBRARY_DIMS,
    CaseAssetResolver,
    ResolvedAssets,
    _read_deck_flex,
    validate_reload_deck,
)
from .genome import depth2_edges_for_fresh_units


#: Stable deck-knob signature for produce records (plan 4.2 record_id preimage).
PRODUCE_DECK_KNOBS = "ga80_produce"


class MissingCaseAssetError(RuntimeError):
    """A verify entry cannot be staged because its restart and/or template deck
    did not resolve.  Raised (instead of silently substituting a ``Path('.')``
    sentinel that the vendor would then ``read_bytes()``/``copy2()`` on the CWD)
    so the entry becomes a CLEAN, descriptive ``error`` outcome — the Bug A root
    cause was the free-search reaching a pair with no package deck and the sentinel
    surfacing as a cryptic ``PermissionError: [Errno 13] Permission denied: '.'``.
    """

#: ``error``-outcome failure tags that are HONEST NEGATIVE LABELS — a physical
#: property of the drawn ``(pattern, restart)`` pair, not a harness defect (plan
#: 5.4 / 4.6 정직 회계).  ``non_finite_flux`` is the NaN watchdog killing a
#: divergent flux solve; future physics-kill tags (e.g. an explicit MASTER
#: divergence code) are added here.  A failure in this set is counted as
#: ``nonfinite`` (treated like ``nonconverged`` for QC), never as a
#: ``harness_error`` that halts a stratum.
PHYSICS_KILL_FAILURES = frozenset({"non_finite_flux"})

#: NaN watchdog defaults: poll each running MASTER's MAS_OUT this often (s) and
#: kill it once its tail shows this many consecutive non-finite iteration lines.
NAN_WATCHDOG_POLL_S = 10.0
NAN_WATCHDOG_STREAK = 12

_NAN_TOKEN_RE = re.compile(r"\bnan\b", re.IGNORECASE)


def _tail_lines(path: Path, *, max_bytes: int = 32768) -> list[str]:
    """Last chunk of ``path`` as lines (memory-bounded; reads only the tail)."""

    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return []
    return data.decode("latin-1", errors="replace").splitlines()


def _mas_out_shows_nan(path: Path, *, streak: int) -> bool:
    """True when the last ``streak`` non-blank lines of ``path`` are all NaN.

    This is the divergent multigroup-outer signature MASTER emits when a loading
    pattern makes the flux eigen-solve non-finite: ``MGOUTER .. NaN NaN NaN``
    repeated endlessly, with no ``BURNUP``/``EFPD`` progress.
    """

    lines = [ln for ln in _tail_lines(path) if ln.strip()]
    if len(lines) < streak:
        return False
    return all(_NAN_TOKEN_RE.search(ln) for ln in lines[-streak:])


# --------------------------------------------------------------------------- #
# MASTER runner with a non-finite-flux watchdog
# --------------------------------------------------------------------------- #
class WatchdogMasterRunner(MasterRunner):
    """:class:`MasterRunner` + a NaN watchdog on the running subprocess.

    The vendor runner (which we must not modify) launches MASTER and blocks in
    ``process.wait(timeout)``.  A divergent pattern floods ``MAS_OUT`` with
    non-finite iteration lines and would otherwise burn the whole ``timeout``.
    We hook the one seam the vendor calls right after ``Popen`` and before the
    wait — :meth:`_apply_cpu_affinity` — to launch a daemon thread that polls the
    active worker ``MAS_OUT`` tail; on the divergence signature it drops a
    ``NONFINITE_FLUX`` sentinel and kills the subprocess, so the wait returns a
    non-zero status and the vendor raises ``MasterRunError`` (mapped by the
    verifier to an honest ``error`` / ``failure="non_finite_flux"`` label — plan
    5.4).  Setup never raises: a watchdog problem must not abort a real run.
    """

    def __init__(
        self,
        *args: Any,
        nan_poll_s: float = NAN_WATCHDOG_POLL_S,
        nan_streak: int = NAN_WATCHDOG_STREAK,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._nan_poll_s = max(0.05, float(nan_poll_s))
        self._nan_streak = max(2, int(nan_streak))

    def _apply_cpu_affinity(self, process: subprocess.Popen[bytes]) -> dict[str, Any]:
        info = super()._apply_cpu_affinity(process)
        try:
            threading.Thread(
                target=self._nan_watchdog,
                args=(process,),
                name="master-nan-watchdog",
                daemon=True,
            ).start()
        except Exception:  # noqa: BLE001 — watchdog setup must never abort a run
            pass
        return info

    def _active_mas_out(self) -> Path | None:
        """The most recently written ``MAS_OUT`` under this worker's work root."""

        try:
            candidates = [p for p in self.work_root.glob("*/MAS_OUT") if p.is_file()]
        except OSError:
            return None
        if not candidates:
            return None
        try:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except OSError:
            return candidates[-1]

    def _nan_watchdog(self, process: subprocess.Popen[bytes]) -> None:
        while process.poll() is None:
            time.sleep(self._nan_poll_s)
            out = self._active_mas_out()
            if out is None:
                continue
            if _mas_out_shows_nan(out, streak=self._nan_streak):
                try:
                    (out.parent / "NONFINITE_FLUX").write_text(
                        "non_finite_flux\n", encoding="utf-8"
                    )
                except OSError:
                    pass
                try:
                    process.kill()
                except OSError:
                    pass
                return


# --------------------------------------------------------------------------- #
# immediate intermediate-cycle purge
# --------------------------------------------------------------------------- #
# USER DIRECTIVE: "평형주기까지 계산하는 하위 주기들의 계산 데이터는 바로바로 삭제하고,
# 평형주기 결과값만 학습 데이터셋으로 남겨" — delete the pre-equilibrium (intermediate)
# cycle calculation artifacts IMMEDIATELY; only the converged equilibrium-cycle
# result stays (it is already what the store keeps).  See PurgingEquilibriumRunner.

#: Files kept in a FAILED chain's retained work dir for error diagnosis (plan 5.4
#: 정직 회계).  Everything else — the large staged libraries (MAS_XSL / MAS_HFF) and
#: every restart (MAS_RST.*), summary (MAS_SUM), and PPI (MAS_PPI.*) — is deleted so
#: the diagnostic tail stays byte-cheap.  ``NONFINITE_FLUX`` (the watchdog sentinel)
#: and ``MAS_OUT`` (its NaN tail) are the EXACT signals :meth:`WaveVerifier._classify_failure`
#: reads to label a physics kill, so they MUST survive the trim.
_FAILED_DIR_KEEP = frozenset(
    {
        "MAS_OUT",
        "MAS_INP",
        "NONFINITE_FLUX",
        "MASTER.stdout",
        "MASTER.stderr",
        "CPU_AFFINITY.json",
    }
)


def _is_strict_subpath(path: Path | None, root: Path | None) -> bool:
    """True iff ``path`` resolves to a STRICT descendant of ``root``.

    Never true for ``path == root``, for a ``None`` on either side, or when either
    path cannot be resolved.  This is the Bug A guard: every purge delete is
    fenced INSIDE the worker's own ``work_root`` so a stray ``Path('')`` /
    ``Path('.')`` / the process CWD / a filesystem anchor can never be removed —
    a candidate whose assets did not resolve must NEVER cost a delete outside the
    per-worker sandbox.
    """

    if path is None or root is None:
        return False
    try:
        resolved_path = Path(path).resolve()
        resolved_root = Path(root).resolve()
    except OSError:
        return False
    if resolved_path == resolved_root:
        return False
    return resolved_root in resolved_path.parents


def _purge_touches(target: Path | None, protected: Path | None) -> bool:
    """True iff ``rmtree(target)`` would remove ``protected`` or any part of it.

    Refuses every overlap on the same root→leaf line: ``target == protected``
    (the delete IS the protected dir), ``target`` an ANCESTOR of ``protected``
    (the recursive delete would take it), or ``target`` a DESCENDANT of
    ``protected`` (deleting a staged case dir or its contents).  Disjoint trees
    return ``False``.

    This is the round-2 ``cases_dir`` fence (bug B defense): the intermediate-
    cycle purge only ever needs to delete per-cycle work dirs under the worker
    ``work_root``.  It must NEVER touch the staged ``produce_cases`` tree, whose
    per-case ``case_dir`` is exactly what the vendor ``master.py:_assets``
    requires to exist ("case directory does not exist").  A stray purge target
    that resolves onto the cases tree — however it arose under high (56-way)
    parallelism — is refused here even when it would satisfy the ``work_root``
    fence, so a purge can never delete a case dir out from under a pending eval.
    """

    if target is None or protected is None:
        return False
    try:
        t = Path(target).resolve()
        p = Path(protected).resolve()
    except OSError:
        return False
    if t == p:
        return True
    return p in t.parents or t in p.parents


def _safe_rmtree(
    path: Path | None, *, work_root: Path | None = None, protect: Path | None = None
) -> None:
    """Best-effort recursive delete; a purge failure must never sink a run.

    Bug A guard: when ``work_root`` is given, the delete is REFUSED (log-and-skip,
    never attempted) unless ``path`` is a strict subpath of it — so a purge can
    never touch ``.``/CWD/a path outside the worker sandbox.  With no ``work_root``
    (the already-safe direct call sites) CWD and a filesystem anchor are still
    refused as a last-resort net.

    Bug B fence: when ``protect`` (the staged ``cases_dir``) is given, the delete
    is additionally REFUSED if it would touch that tree (see :func:`_purge_touches`).
    """

    if path is None:
        return
    target = Path(path)
    if protect is not None and _purge_touches(target, protect):
        return  # never touch the staged cases_dir tree (round-2 bug B fence)
    if work_root is not None:
        if not _is_strict_subpath(target, work_root):
            return  # outside the per-worker sandbox — refuse, do not delete
    else:
        try:
            resolved = target.resolve()
        except OSError:
            return
        if resolved == Path.cwd().resolve() or resolved == Path(resolved.anchor):
            return  # never the CWD or a filesystem root, even unguarded
    try:
        shutil.rmtree(target)
    except OSError:
        # A retained artifact is preferable to hiding a real result behind a
        # cleanup-only failure (matches the vendor EquilibriumRunner._clean).
        pass


def _trim_failed_work_dir(
    work_dir: Path | None,
    *,
    work_root: Path | None = None,
    protect: Path | None = None,
) -> None:
    """Shrink a retained FAILED-cycle dir to a small diagnostic tail.

    Keeps only :data:`_FAILED_DIR_KEEP` (the watchdog sentinel, a MAS_OUT tail,
    the deck, and stdio — all the error-diagnosis harness needs) and removes the
    big staged libraries and every ``MAS_RST.*`` / ``MAS_SUM`` / ``MAS_PPI.*`` so a
    divergent or broken chain leaves a byte-cheap footprint, not a full staged
    case.  Never raises: trimming is a courtesy, not a correctness requirement.

    Bug A guard: when ``work_root`` is given, trimming is REFUSED unless
    ``work_dir`` is a strict subpath of it, so a ``Path('.')`` / CWD is never
    emptied of its contents.
    """

    if work_dir is None:
        return
    if protect is not None and _purge_touches(work_dir, protect):
        return  # never trim inside the staged cases_dir tree (round-2 bug B fence)
    if work_root is not None and not _is_strict_subpath(work_dir, work_root):
        return  # outside the per-worker sandbox — refuse to trim
    try:
        entries = list(Path(work_dir).iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name in _FAILED_DIR_KEEP:
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError:
            pass


class PurgingEquilibriumRunner(EquilibriumRunner):
    """:class:`EquilibriumRunner` that deletes each intermediate cycle's work
    products the instant the NEXT cycle no longer needs them (USER DIRECTIVE).

    The vendor runner chains cycles and only bulk-deletes every retained work dir
    at CHAIN END (``keep_success=False``) or never (``keep_success=True``), so a
    14-cycle chain piles up to 14 fully staged cases at peak.  The vendor exposes
    no per-cycle callback, but it DOES call :meth:`_generated_restart` exactly once
    per cycle — right after that cycle's MASTER run and before the next — so we
    override that seam (delegating to ``super()`` for the real work) to rmtree the
    PREVIOUS cycle's dir the moment its restart has been consumed by the current
    cycle.  This achieves TRUE per-cycle immediate deletion (peak ≈ 2 dirs), not a
    chain-end purge, without editing any vendor logic.

    Survivors — nothing else is kept:

    * the FINAL equilibrium cycle's dir (MAS_SUM + the final MAS_RST.*) is left
      untouched, so the vendor's own ``keep_success`` decides its fate: kept for
      harvest / CaseAssetResolver promotion / bootstrap band-seed when ``True``,
      cleaned by the vendor at chain end when ``False``;
    * a FAILED chain's last (failing) cycle dir, trimmed by
      :func:`_trim_failed_work_dir` to the NONFINITE sentinel / MAS_OUT tail the
      QC classifier reads.

    ``purge_intermediate=False`` restores the exact vendor lifecycle (no hook).
    """

    def __init__(
        self,
        *args: Any,
        purge_intermediate: bool = True,
        protect_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._purge_intermediate = bool(purge_intermediate)
        #: The staged ``cases_dir`` tree the purge must NEVER touch (bug B fence);
        #: every per-case ``case_dir`` (what ``master.py:_assets`` requires to
        #: exist) lives under it.  ``None`` restores the pre-fence behaviour.
        self._protect_dir: Path | None = (
            Path(protect_dir) if protect_dir is not None else None
        )
        #: The most-recent cycle's work dir, tracked WITHIN a single chain so a
        #: prior candidate's kept final dir (keep_success=True) is never touched.
        self._prev_cycle_dir: Path | None = None

    def _guard_root(self) -> Path | None:
        """The worker ``work_root`` every purge delete must stay strictly inside.

        Bug A fence: cycle work dirs are ``tempfile.mkdtemp(dir=work_root)`` — true
        strict subpaths — so this admits every legitimate cycle dir while refusing
        ``.``/CWD/anything outside the sandbox.
        """

        root = getattr(self.master_runner, "work_root", None)
        return Path(root) if root is not None else None

    def run(self, case_data: CaseData, pattern: Pattern) -> Any:
        # Per-chain reset FIRST: a fresh chain must not inherit the previous
        # candidate's final dir as a purgeable "previous cycle".
        self._prev_cycle_dir = None
        if not self._purge_intermediate:
            return super().run(case_data, pattern)
        try:
            return super().run(case_data, pattern)
        except MasterRunError as exc:
            # A failed chain (EquilibriumRunError is a MasterRunError) retains its
            # failing cycle dir for diagnosis; trim it to a small tail.
            _trim_failed_work_dir(
                getattr(exc, "work_dir", None),
                work_root=self._guard_root(),
                protect=self._protect_dir,
            )
            raise

    # ``EquilibriumRunner`` aliases ``evaluate = run``; rebind so both entry
    # points get the per-chain reset + failed-dir trim.
    evaluate = run

    def _generated_restart(self, work_dir: Path, input_restart: Path) -> Path:
        generated = super()._generated_restart(work_dir, input_restart)
        if self._purge_intermediate:
            # The current cycle's run already staged (copied) its input restart
            # into ``work_dir``, so the PREVIOUS cycle's dir is now fully consumed
            # and can be deleted immediately.  (One cycle of lag is unavoidable:
            # the prior dir cannot go until the current cycle has read from it.)
            prev = self._prev_cycle_dir
            if prev is not None and prev != work_dir:
                _safe_rmtree(
                    prev, work_root=self._guard_root(), protect=self._protect_dir
                )
            self._prev_cycle_dir = work_dir
        return generated


# --------------------------------------------------------------------------- #
# EDIT5 map harvesting (forensic 20260723)
# --------------------------------------------------------------------------- #
def _maps_from_equilibrium_result(result: Any) -> Any:
    """Best-effort EDIT5 map stack ``(4,9,9)`` from a converged raw EquilibriumResult.

    Reads the final cycle's ``MAS_SUM`` — which survives because ``harvest_maps``
    forces ``keep_success`` — from ``result.cycles[-1].work_dir`` (or any
    ``result.retained_work_dirs``).  MUST be given the RAW ``EquilibriumResult``
    (with ``.cycles``); the vendor ``EvaluationResult`` drops it, which is why the
    harvest happens in :class:`HarvestingEquilibriumEvaluator` and not downstream.
    Returns ``None`` on ANY failure — a missing/stale dir or a parse gap must never
    abort a wave (the F_r label is valid without its map)."""
    try:
        from ..data.edit5 import parse_mas_sum, stack_maps
        cycles = getattr(result, "cycles", None)
        candidates: list[Path] = []
        if cycles:
            wd = getattr(cycles[-1], "work_dir", None)
            if wd is not None:
                candidates.append(Path(wd))
        for wd in (getattr(result, "retained_work_dirs", None) or ()):
            candidates.append(Path(wd))
        for wd in candidates:
            sum_path = wd / "MAS_SUM"
            if sum_path.is_file():
                summary = parse_mas_sum(sum_path)
                if getattr(summary, "edit5_maps", None):
                    return stack_maps(summary)
    except Exception:  # noqa: BLE001 — maps are optional; never abort a wave
        return None
    return None


def _hires_from_equilibrium_result(result: Any) -> dict[str, Any] | None:
    """High-resolution harvest ``{"traj": (n_steps,3,9,9), "axial": (n_steps,25)}``.

    Everything here is ALREADY parsed by ``parse_mas_sum`` for the (4,9,9) stack —
    the legacy path just threw ~28 of the ~30 burnup steps and all of EDIT 6 away.
    Harvesting them costs ~1 ms/record of stacking and ~14 KiB/record of float16
    storage, and is NOT retroactive: a record produced without it loses the
    resolution permanently.  Same never-abort contract as the (4,9,9) harvest —
    any failure returns ``None`` and the F_r/cyclen labels stand on their own.
    """
    try:
        from ..data.edit5 import parse_mas_sum, stack_axial, stack_step_maps
        cycles = getattr(result, "cycles", None)
        candidates: list[Path] = []
        if cycles:
            wd = getattr(cycles[-1], "work_dir", None)
            if wd is not None:
                candidates.append(Path(wd))
        for wd in (getattr(result, "retained_work_dirs", None) or ()):
            candidates.append(Path(wd))
        for wd in candidates:
            sum_path = wd / "MAS_SUM"
            if not sum_path.is_file():
                continue
            summary = parse_mas_sum(sum_path)
            out: dict[str, Any] = {}
            if getattr(summary, "edit5_maps", None):
                out["traj"] = stack_step_maps(summary)
            if getattr(summary, "axial_rows", None):
                out["axial"] = stack_axial(summary)
            if out:
                return out
    except Exception:  # noqa: BLE001 — optional; never abort a wave
        return None
    return None


def _eq_provenance(result: Any) -> dict[str, str] | None:
    """Branch assets of a converged chain: the final cycle's deck + its restart.

    The SDM/MTC pre-delivery gate (decision D9, :mod:`.sdm_mtc`) has to rebuild
    ``%EXE_RHO`` / ``%EXE_ROD`` branch decks from **this candidate's own**
    converged equilibrium state.  Both halves exist for exactly one moment — the
    final cycle's ``work_dir`` holds its ``MAS_INP`` and the restart that cycle
    generated, and the purge/lifecycle deletes them as soon as the wave moves on —
    so the paths are captured here, where the raw ``EquilibriumResult`` (with
    ``.cycles``) is still in scope, exactly like the map harvest above.

    Returns ``None`` on any failure, and only returns paths that are on disk RIGHT
    NOW: a half-resolved entry would send the gate looking for a restart that a
    ``keep_success=False`` verifier already deleted, and the honest outcome there
    is "unverifiable", not "verified against something else".
    """
    try:
        cycles = getattr(result, "cycles", None)
        if not cycles:
            return None
        final = cycles[-1]
        work_dir = getattr(final, "work_dir", None)
        restart = getattr(final, "restart_path", None)
        if work_dir is None or restart is None:
            return None
        deck = Path(work_dir) / "MAS_INP"
        restart = Path(restart)
        if not deck.is_file() or not restart.is_file():
            return None
        return {"deck": str(deck), "restart": str(restart),
                "work_dir": str(work_dir)}
    except Exception:  # noqa: BLE001 — provenance is optional; never abort a wave
        return None


class HarvestingEquilibriumEvaluator:
    """:class:`EquilibriumEvaluator` that also harvests the converged EDIT5 map.

    The vendor evaluator's ``EvaluationResult`` carries only ``fom`` + ``metadata``
    (the raw ``EquilibriumResult.cycles`` — and thus the MAS_SUM work dir — is
    dropped), so the map has to be captured HERE, where ``runner.run`` returns the
    raw result.  It is passed downstream via ``metadata["maps"]`` (an additive key
    every existing consumer ignores).  A non-converged run harvests nothing."""

    def __init__(self, equilibrium_runner: Any) -> None:
        from ..vendor.masterrl.search import EquilibriumEvaluator
        self._inner = EquilibriumEvaluator(equilibrium_runner)
        self.runner = equilibrium_runner

    def evaluate(self, case: Any, pattern: Any) -> Any:
        from dataclasses import replace as _rp
        result = self.runner.run(case, pattern)                 # raw EquilibriumResult
        # Rebuild the vendor EvaluationResult via the inner adapter's own logic by
        # re-deriving from the SAME raw result (no second run): mirror its fields,
        # then attach the harvested map.
        base = self._inner_evaluation(result)
        if bool(getattr(result, "converged", False)):
            maps = _maps_from_equilibrium_result(result)
            if maps is not None:
                base = _rp(base, metadata={**base.metadata, "maps": maps})
            # High-resolution siblings (all-burnup-step EDIT5 + EDIT6 axial).
            # Additive metadata key: every existing consumer ignores it.
            hires = _hires_from_equilibrium_result(result)
            if hires:
                base = _rp(base, metadata={**base.metadata, "maps_hires": hires})
            # SDM/MTC branch assets (decision D9) — additive metadata key.
            provenance = _eq_provenance(result)
            if provenance:
                base = _rp(base, metadata={**base.metadata,
                                           "eq_provenance": provenance})
        return base

    def _inner_evaluation(self, result: Any) -> Any:
        """Build the vendor ``EvaluationResult`` from an already-run raw result
        (byte-identical to :meth:`EquilibriumEvaluator.evaluate`'s construction, but
        without re-running the chain)."""
        from dataclasses import replace as _rp
        from ..vendor.masterrl.search import EvaluationResult
        return EvaluationResult(
            fom=_rp(result.fom, converged=bool(result.converged)),
            raw_master_calls=int(result.master_process_calls),
            metadata={
                "mode": "equilibrium_master",
                "converged": bool(result.converged),
                "n_cycles": int(result.n_cycles),
                "comparisons": [
                    {
                        "previous_cycle": c.previous_cycle,
                        "current_cycle": c.current_cycle,
                        "deltas": dict(c.deltas),
                        "within_tolerance": c.within_tolerance,
                    }
                    for c in result.comparisons
                ],
                "tolerances": result.tolerances.as_dict(),
            },
        )


# --------------------------------------------------------------------------- #
# wave entry / outcome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WaveEntry:
    """One candidate to verify: pattern + its resolved case assets + metadata."""

    pattern: Pattern
    case_key: CaseKey
    resolved_assets: ResolvedAssets
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WaveOutcome:
    """One verified candidate result (budget-honest three-way taxonomy)."""

    status: str                    # "converged" | "nonconverged" | "error"
    fom: FOM | None
    n_cycles: int
    tolerance_margin: float | None
    wall_s: float
    restart_provenance: str
    failure: str
    converged_at_cap: bool
    case_key: CaseKey
    pattern: Pattern
    meta: dict[str, Any] = field(default_factory=dict)
    #: Harvested EDIT5 assembly-map stack ``(4, 9, 9)`` float32 (boc/eoc power +
    #: eoc burnup/kinf) for a converged candidate when the verifier was built with
    #: ``harvest_maps=True``; ``None`` otherwise.  Written to the store's maps.npz
    #: keyed by record_id so the node-power model has campaign-label maps.
    maps: Any = None
    #: High-resolution siblings of :attr:`maps`, ``{"traj": (n_steps,3,9,9),
    #: "axial": (n_steps,25)}`` — the full EDIT5 burnup trajectory and the EDIT6
    #: axial shape, both already parsed for ``maps`` and previously discarded.
    #: Stored under ``<record_id>__traj`` / ``<record_id>__axial`` keys so the
    #: legacy ``<record_id>`` -> (4,9,9) contract is untouched.
    maps_hires: Any = None
    #: CPU class the chain ran on — ``"P"`` (performance), ``"E"`` (efficiency),
    #: or ``""`` (unknown / unpinned / stub).  Accounting only (E-core chains run
    #: ~30-40% slower); it distinguishes ``wall_s`` by class but drives NO
    #: behaviour — plan section 5.4 observability for the all-cores production mix.
    core_class: str = ""
    #: ``{"deck", "restart", "work_dir"}`` of the CONVERGED final cycle, when the
    #: verifier retained it (:func:`_eq_provenance`).  The SDM/MTC pre-delivery
    #: gate (decision D9) rebuilds this candidate's branch decks from exactly these
    #: two files; ``None`` means the candidate is honestly unverifiable, never that
    #: another candidate's restart may be substituted.
    eq_provenance: dict[str, str] | None = None


def classify_outcome(outcome: WaveOutcome) -> str:
    """Map a :class:`WaveOutcome` to its QC counter class (plan 5.4).

    Four classes refine the three-way ``status`` for stratum accounting:

    * ``converged``     — a converged equilibrium label;
    * ``nonconverged``  — an honest non-convergence (cap-exhausted, etc.);
    * ``nonfinite``     — an ``error`` whose ``failure`` is a physics kill
      (:data:`PHYSICS_KILL_FAILURES`, e.g. ``non_finite_flux``): an HONEST
      NEGATIVE label about the pattern, NOT a harness fault;
    * ``harness_error`` — any other ``error`` (staging / deck / resolver /
      timeout / genome defect): a real harness fault that must halt a stratum.

    Only ``harness_error`` participates in the HALT rule; ``nonfinite`` groups
    with ``nonconverged`` for the (non-halting) rebalancing advisory.
    """

    if outcome.status == "converged":
        return "converged"
    if outcome.status == "nonconverged":
        return "nonconverged"
    # status == "error"
    if outcome.failure in PHYSICS_KILL_FAILURES:
        return "nonfinite"
    return "harness_error"


# --------------------------------------------------------------------------- #
# verifier
# --------------------------------------------------------------------------- #
EvaluatorFactory = Callable[[int, int | None], Any]


class WaveVerifier:
    """Fixed-size MASTER wave verifier (ported ``_ga_case_evaluator`` wiring)."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        package_root: str | Path | None = None,
        executable: str | Sequence[str] | None = None,
        workers: int | None = None,
        timeout: float = 3600.0,
        max_cycles: int = 14,
        consecutive: int = 2,
        tolerances: Mapping[str, float | None] | None = None,
        cache_dir: str | Path | None = None,
        keep_success: bool = False,
        evaluator_factory: EvaluatorFactory | None = None,
        resolver: CaseAssetResolver | None = None,
        stage_decks: bool | None = None,
        nan_poll_s: float = NAN_WATCHDOG_POLL_S,
        nan_streak: int = NAN_WATCHDOG_STREAK,
        library_dims: tuple[int, int] = LIBRARY_DIMS,
        use_all_cores: bool = False,
        host_reserve: int = 1,
        assign_cores: bool | None = None,
        purge_intermediate: bool = True,
        harvest_maps: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.package_root = Path(package_root) if package_root is not None else None
        self.executable = executable
        self.timeout = float(timeout)
        self.max_cycles = int(max_cycles)
        self.consecutive = int(consecutive)
        self.tolerances = tolerances
        #: Harvest converged EDIT5 maps into the store (forces keep_success so the
        #: final cycle's MAS_SUM survives to :meth:`_result_to_outcome`).
        self.harvest_maps = bool(harvest_maps)
        self.keep_success = bool(keep_success) or self.harvest_maps
        #: Delete each intermediate cycle's work products as soon as the next
        #: cycle no longer needs them (USER DIRECTIVE).  Honoured by the default
        #: (real-MASTER) evaluator factory; an injected factory must build its own
        #: :class:`PurgingEquilibriumRunner` to opt in (curriculum does).
        self.purge_intermediate = bool(purge_intermediate)
        self.nan_poll_s = float(nan_poll_s)
        self.nan_streak = int(nan_streak)
        self.library_dims = tuple(library_dims)
        self.use_all_cores = bool(use_all_cores)
        self.host_reserve = max(0, int(host_reserve))
        self._factory = evaluator_factory
        self.resolver = resolver or CaseAssetResolver(self.package_root or ".")
        self.cases_dir = self.run_dir / "produce_cases"
        self.work_root = self.run_dir / "master_work"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.run_dir / "master_cache"

        # Worker / core policy (ported from flow.py L272-308).  ``assign_cores``
        # decouples *physical core assignment* from *who supplies the evaluator*:
        # a plain stub run wants neither detection nor pinning, but the curriculum
        # injects a REAL pin-burnup MASTER factory that DOES want P/E assignment,
        # so it passes ``assign_cores=True`` (default: assign iff no injected
        # factory — the historic behaviour).
        assign = assign_cores if assign_cores is not None else (evaluator_factory is None)
        self.worker_cores: list[int | None]
        self.host_cores: tuple[int, ...] | None
        self.worker_core_class: list[str]
        if not assign:
            # Injected stub evaluator: no core detection, no deck staging.
            self.n_workers = max(1, int(workers) if workers else 4)
            self.worker_cores = [None] * self.n_workers
            self.host_cores = None
            self.worker_core_class = [""] * self.n_workers
            self.stage_decks = bool(stage_decks) if stage_decks is not None else False
        else:
            layout = detect_core_layout()
            performance = list(layout.performance)
            efficiency = list(layout.efficiency)
            perf_set = set(performance)
            if self.use_all_cores:
                # DIRECTIVE: fill EVERY logical core — P-cores FIRST, then E-cores,
                # each worker pinned 1:1 — minus ``host_reserve`` core(s) the host
                # retreats to while the wave owns the CPU.  ``workers`` still caps.
                ordered = performance + efficiency
                total = len(ordered)
                reserve = min(self.host_reserve, max(0, total - 1))  # keep >=1 worker
                pool = ordered[: total - reserve] if reserve else list(ordered)
                reserved = list(ordered[total - reserve:]) if reserve else []
                requested = int(workers) if workers else len(pool)
                self.n_workers = max(1, min(requested, len(pool)))
                self.worker_cores = list(pool[: self.n_workers])
                # Host = the reserved core(s) plus any pool cores an explicit
                # ``workers`` cap left idle (never a worker core).
                leftover = list(pool[self.n_workers:])
                self.host_cores = tuple(reserved + leftover) or None
            else:
                # Legacy: P-cores only; the host retreats to the E-cores.
                requested = int(workers) if workers else len(performance)
                self.n_workers = max(1, min(requested, len(performance)))
                self.worker_cores = list(performance[: self.n_workers])
                self.host_cores = layout.efficiency or None
            self.worker_core_class = [
                "P" if core in perf_set else ("E" if core is not None else "")
                for core in self.worker_cores
            ]
            self.stage_decks = bool(stage_decks) if stage_decks is not None else True

        self.evaluators: list[Any] | None = None

    # -- lazy worker construction ------------------------------------------ #
    def _default_factory(self, worker_id: int, cpu_core: int | None) -> Any:
        """Build one per-worker MasterRunner -> EquilibriumRunner -> Evaluator."""

        from ..vendor.masterrl.search import EquilibriumEvaluator

        if self.package_root is None or self.executable is None:
            raise RuntimeError(
                "the default (real MASTER) evaluator requires package_root and "
                "executable; inject an evaluator_factory for dry runs/tests"
            )
        master = WatchdogMasterRunner(
            self.package_root,
            self.executable,
            work_root=self.work_root / f"worker_{worker_id:02d}",
            cache_dir=self.cache_dir / f"worker_{worker_id:02d}",
            timeout=self.timeout,
            keep_success=self.keep_success,
            cpu_core=cpu_core,
            nan_poll_s=self.nan_poll_s,
            nan_streak=self.nan_streak,
        )
        equilibrium = PurgingEquilibriumRunner(
            master,
            max_cycles=self.max_cycles,
            consecutive=self.consecutive,
            tolerances=self.tolerances,
            keep_success=self.keep_success,
            enable_pin_burnup=False,
            purge_intermediate=self.purge_intermediate,
            protect_dir=self.cases_dir,
        )
        if self.harvest_maps:
            return HarvestingEquilibriumEvaluator(equilibrium)
        return EquilibriumEvaluator(equilibrium)

    def _ensure_evaluators(self) -> list[Any]:
        if self.evaluators is None:
            factory = self._factory or self._default_factory
            self.evaluators = [
                factory(worker_id, self.worker_cores[worker_id])
                for worker_id in range(self.n_workers)
            ]
        return self.evaluators

    # -- per-entry case data ----------------------------------------------- #
    def _build_case_data(self, entry: WaveEntry) -> CaseData:
        resolved = entry.resolved_assets
        cell = float(entry.meta.get("e_core") or 0.0)

        # Bug A root cause: the free-search reaches pairs with no package
        # template deck (and/or restart).  On the REAL-MASTER path (stage_decks),
        # the vendor will ``read_bytes()`` the template and ``copy2()`` the restart,
        # so a missing asset MUST become a clean, descriptive ``error`` here — never
        # a ``Path('.')`` sentinel that the vendor reads as the CWD directory and
        # surfaces as ``PermissionError: [Errno 13] Permission denied: '.'``.  Stub
        # / dry runs (stage_decks=False) never touch the deck, so the harmless
        # sentinel is preserved for them.
        if self.stage_decks and (
            resolved.template_deck_path is None or resolved.restart_path is None
        ):
            missing = []
            if resolved.template_deck_path is None:
                missing.append("template deck")
            if resolved.restart_path is None:
                missing.append("restart")
            raise MissingCaseAssetError(
                f"cannot stage a verification deck for {entry.case_key.label}: "
                f"no {' and no '.join(missing)} resolved "
                f"(restart_provenance={resolved.restart_provenance!r}); this pair "
                "is not reachable in the package — inject a store elite / add a "
                "template+restart, or drop the pair from the search universe"
            )

        template_path = resolved.template_deck_path or Path(".")
        restart_path = resolved.restart_path or Path(".")

        # Per-library (paramA) alias space for the vendor runner: it re-injects
        # ``pattern.to_shf()`` and validates ``pattern.validate_case(key.pair, ...)``
        # on EVERY cycle, so the case key handed to it must share the (alias) space
        # of the pattern the evaluator runs (see ``_eval_entry``).  ga80 -> no-op.
        run_key = self.resolver.alias_case_key(entry.case_key)

        if (
            self.stage_decks
            and resolved.template_deck_path is not None
            and resolved.restart_path is not None
        ):
            template_text = _read_deck_flex(resolved.template_deck_path)
            prepared = self.resolver.prepare_cycle1_deck(
                template_text, entry.pattern, resolved.restart_path.name
            )
            # Pre-Popen sanity gate: a malformed reload deck (missing depletion
            # chain / EOC restart write, stray fresh-core batch map, wrong restart
            # reference, or wrong library dims) drives MASTER into a NaN loop.
            # Refuse it here — a clean error, never a launched divergence.
            validate_reload_deck(
                prepared, resolved.restart_path.name, expected_dims=self.library_dims
            )
            cycle = deck_cycle(prepared)
            # Key the staged case dir by (pattern, restart): the SAME pattern
            # evaluated against different restarts (cross-restart fixed-point
            # trials, restart-provenance production) must stage to distinct dirs,
            # or two concurrent workers race on one deck file and each other's
            # restart reference (plan 5.2).
            restart_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved.restart_path.name)
            case_dir = (
                self.cases_dir / entry.case_key.folder
                / f"{entry.pattern.digest}__{restart_tag}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            deck_path = case_dir / f"MAS_INP_cy{cycle:02d}.inp"
            deck_path.write_bytes(prepared.encode("utf-8"))
            template_path = deck_path

        return CaseData(
            key=run_key,
            cell=cell,
            records=(),
            template_path=template_path,
            restart_path=restart_path,
        )

    # -- evaluation -------------------------------------------------------- #
    def evaluate_wave(self, entries: Sequence[WaveEntry]) -> list[WaveOutcome]:
        """Verify a wave; one worker per entry (chunked by worker count)."""

        if not entries:
            return []
        evaluators = self._ensure_evaluators()
        outcomes: list[WaveOutcome | None] = [None] * len(entries)

        with host_affinity(self.host_cores):
            with ThreadPoolExecutor(
                max_workers=self.n_workers, thread_name_prefix="verify-worker"
            ) as pool:
                for start in range(0, len(entries), self.n_workers):
                    chunk = list(entries[start : start + self.n_workers])
                    futures = []
                    for offset, entry in enumerate(chunk):
                        evaluator = evaluators[offset]
                        core_class = self.worker_core_class[offset]
                        # Deck prep + staging happen *inside* the worker future so
                        # a per-entry resolver/prep/validation failure becomes that
                        # entry's own ``error`` outcome and never aborts the wave
                        # (plan 5.4 — one bad candidate must not sink its 7 peers).
                        future = pool.submit(
                            self._eval_entry, evaluator, entry, core_class
                        )
                        futures.append((start + offset, future))
                    for position, future in futures:
                        outcomes[position] = future.result()
        return [outcome for outcome in outcomes if outcome is not None]

    def _eval_entry(
        self, evaluator: Any, entry: WaveEntry, core_class: str = ""
    ) -> WaveOutcome:
        start = time.perf_counter()
        try:
            case_data = self._build_case_data(entry)
            # Hand the runner an alias-space pattern matched to ``case_data.key``
            # (paramA); ga80 -> unchanged.  The store/outcome keep ``entry`` (the
            # type_id pattern/key) so featurization + records stay in type_id space.
            run_pattern = self.resolver.alias_pattern(entry.pattern)
            result = evaluator.evaluate(case_data, run_pattern)
        except Exception as exc:  # noqa: BLE001 — in-place failure = an "error" label
            wall = time.perf_counter() - start
            return WaveOutcome(
                status="error",
                fom=None,
                n_cycles=0,
                tolerance_margin=None,
                wall_s=wall,
                restart_provenance=entry.resolved_assets.restart_provenance,
                failure=self._classify_failure(exc),
                converged_at_cap=False,
                case_key=entry.case_key,
                pattern=entry.pattern,
                meta=dict(entry.meta),
                core_class=core_class,
            )
        wall = time.perf_counter() - start
        return self._result_to_outcome(result, entry, wall, core_class)

    def _classify_failure(self, exc: Exception) -> str:
        """Label an evaluation failure; a NaN-diverged run is ``non_finite_flux``.

        A divergent flux solve is an honest negative property of the ``(pattern,
        restart)`` pair, not a harness crash — the watchdog (or a natural timeout)
        leaves a ``NONFINITE_FLUX`` sentinel / NaN ``MAS_OUT`` tail in the retained
        work dir, which the raised ``MasterRunError`` carries via ``work_dir``.
        """

        work_dir = getattr(exc, "work_dir", None)
        if work_dir is not None:
            wd = Path(work_dir)
            try:
                if (wd / "NONFINITE_FLUX").exists():
                    return "non_finite_flux"
                mas_out = wd / "MAS_OUT"
                if mas_out.is_file() and _mas_out_shows_nan(mas_out, streak=self.nan_streak):
                    return "non_finite_flux"
            except OSError:
                pass
        return f"{type(exc).__name__}: {exc}"

    def _result_to_outcome(
        self, result: Any, entry: WaveEntry, wall: float, core_class: str = ""
    ) -> WaveOutcome:
        meta = getattr(result, "metadata", None) or {}
        fom = getattr(result, "fom", None)
        converged = bool(meta.get("converged", getattr(fom, "converged", False)))
        n_cycles = int(meta.get("n_cycles", 0) or 0)
        tolerance_margin = meta.get("tolerance_margin")
        if tolerance_margin is None:
            tolerance_margin = _tolerance_margin_from_meta(meta)
        converged_at_cap = converged and n_cycles > 0 and n_cycles == self.max_cycles
        status = "converged" if converged else "nonconverged"
        # The maps were harvested in HarvestingEquilibriumEvaluator (where the raw
        # EquilibriumResult — with its .cycles work dirs — was in scope; the vendor
        # EvaluationResult drops .cycles), and passed through ``metadata["maps"]``.
        maps = meta.get("maps") if (self.harvest_maps and converged) else None
        maps_hires = (meta.get("maps_hires")
                      if (self.harvest_maps and converged) else None)
        return WaveOutcome(
            status=status,
            fom=fom,
            n_cycles=n_cycles,
            tolerance_margin=tolerance_margin,
            wall_s=wall,
            restart_provenance=entry.resolved_assets.restart_provenance,
            failure="",
            converged_at_cap=converged_at_cap,
            case_key=entry.case_key,
            pattern=entry.pattern,
            meta=dict(entry.meta),
            maps=maps,
            maps_hires=maps_hires,
            core_class=core_class,
            eq_provenance=(meta.get("eq_provenance") if converged else None),
        )

    @staticmethod
    def _extract_maps(result: Any) -> Any:
        """Delegate to :func:`_maps_from_equilibrium_result` (kept as a WaveVerifier
        static method for back-compat / unit tests)."""
        return _maps_from_equilibrium_result(result)


def _tolerance_margin_from_meta(meta: Mapping[str, Any]) -> float | None:
    """Reconstruct :attr:`EquilibriumResult.tolerance_margin` from evaluator meta."""

    comparisons = meta.get("comparisons") or []
    tolerances = meta.get("tolerances") or {}
    if not comparisons:
        return None
    deltas = comparisons[-1].get("deltas", {})
    ratios: list[float] = []
    for name, delta in deltas.items():
        tol = tolerances.get(name)
        if tol is None:
            continue
        if tol > 0.0:
            ratios.append(float(delta) / float(tol))
        else:
            ratios.append(float("inf") if delta > 0.0 else 0.0)
    return max(ratios) if ratios else None


# --------------------------------------------------------------------------- #
# outcome -> CanonicalRecord
# --------------------------------------------------------------------------- #
def outcome_to_record(
    outcome: WaveOutcome,
    *,
    dataset: str = "P",
    library_id: str,
    stratum: str | None = None,
    generator: str | None = None,
    parent_record_id: str | None = None,
    campaign: str | None = None,
    e_core: float | None = None,
    e_split: float | None = None,
    deck_knobs: str = PRODUCE_DECK_KNOBS,
) -> CanonicalRecord:
    """Convert a :class:`WaveOutcome` to a store :class:`CanonicalRecord`.

    ``converged_at_cap`` handling (plan 4.4): a cap-exact convergence is recorded
    as ``converged=True, converged_at_cap=True`` so the model masks its
    convergence label as *unknown* (not a physical negative).  A cap-exhausted
    non-convergence keeps ``converged=False``; the ``(pattern, restart,
    max_cycles)`` context that governs it lives in ``restart_provenance`` +
    ``n_cycles`` (plan 5.4).  An ``error`` outcome yields an all-``None`` target
    row with ``valid=False`` and the failure message retained.
    """

    pattern = outcome.pattern
    case_pair = outcome.case_key.pair
    feed = int(outcome.case_key.feed)
    record_id = compute_record_id(pattern.canonical(), library_id, case_pair, deck_knobs)

    n_fresh = (feed - 1) // 4
    depth2 = (
        depth2_edges_for_fresh_units(n_fresh)
        if (feed - 1) % 4 == 0 and 0 <= n_fresh <= 30
        else 0
    )
    n_batches = 2 if depth2 == 0 else 3

    fom = outcome.fom

    def _f(attr: str) -> float | None:
        return None if fom is None else getattr(fom, attr)

    # Flatness scalars come from the harvested map itself, so they land in the
    # SAME atomic record write as the F_r/cyclen labels — no second pass, no
    # window where a map exists but its scalars do not.  ``record_flatness``
    # never raises: no map (or an odd one) simply leaves both columns null.
    node_peak, map_cov = record_flatness(getattr(outcome, "maps", None))

    return CanonicalRecord(
        record_id=record_id,
        dataset=dataset,
        campaign=campaign,
        stratum=stratum,
        generator=generator,
        parent_record_id=parent_record_id,
        case_pair=case_pair,
        feed=feed,
        n_batches=n_batches,
        depth2_edges=depth2,
        e_core=e_core,
        e_split=e_split,
        library_id=library_id,
        sym_class=SYM_CLASS,
        pattern=pack_pattern(pattern),
        f_r=_f("f_r"),
        f_q=_f("f_q"),
        cbc_max=_f("cbc_max"),
        cbc_boc=None,
        cbc_kind="max",
        cyclen=_f("cyclen"),
        ao_abs=(None if fom is None else fom.ao_abs),
        cycle_burnup=None,
        discharge_burnup=None,
        max_assembly_burnup=(None if fom is None else fom.max_assembly_burnup),
        max_pin_burnup=_f("max_pin_burnup"),
        eoc_ppm=None,
        delta_efpd=None,
        n_cycles=float(outcome.n_cycles) if outcome.n_cycles else None,
        converged=(outcome.status == "converged"),
        converged_at_cap=outcome.converged_at_cap,
        tolerance_margin=outcome.tolerance_margin,
        restart_provenance=outcome.restart_provenance,
        valid=(outcome.status != "error"),
        failure=outcome.failure,
        # maps_key == record_id iff this outcome carries a harvested EDIT5 map
        # stack (verifier built with harvest_maps=True); the caller writes the
        # {record_id: maps} entry to the store's maps.npz.
        maps_key=(record_id if getattr(outcome, "maps", None) is not None else None),
        node_peak=node_peak,
        map_cov=map_cov,
    )


__all__ = [
    "NAN_WATCHDOG_POLL_S",
    "NAN_WATCHDOG_STREAK",
    "MissingCaseAssetError",
    "PHYSICS_KILL_FAILURES",
    "PRODUCE_DECK_KNOBS",
    "PurgingEquilibriumRunner",
    "WatchdogMasterRunner",
    "WaveEntry",
    "WaveOutcome",
    "WaveVerifier",
    "classify_outcome",
    "outcome_to_record",
]
