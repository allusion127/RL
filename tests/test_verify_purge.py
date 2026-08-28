"""Immediate intermediate-cycle purge (USER DIRECTIVE: 평형주기까지 계산하는 하위
주기들의 계산 데이터는 바로바로 삭제하고, 평형주기 결과값만 학습 데이터셋으로 남겨).

Drives a REAL multi-cycle vendor ``EquilibriumRunner`` chain through
``PurgingEquilibriumRunner`` with a tiny fake MASTER that chains restarts and
emits a fixed (converging) MAS_SUM.  Asserts the disk lifecycle the directive
requires:

* a converged, kept chain leaves ONLY the final equilibrium cycle's dir
  (MAS_SUM + final MAS_RST.*), intermediate cycles deleted per-cycle;
* the promotion path still finds that final restart;
* a converged, non-kept chain leaves nothing (vendor cleans the final too);
* a FAILED chain keeps only its failing cycle's dir, trimmed to a small
  diagnostic tail (MAS_OUT + NONFINITE sentinel), big staged files gone;
* ``purge_intermediate=False`` restores the exact vendor lifecycle (all kept);
* the ``[produce]`` config defaults (purge_intermediate=True, consecutive=2).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from lpopt.config import ProduceConfig
from lpopt.search.assets import CaseAssetResolver
from lpopt.search.genome import random_genome
from lpopt.search.verify import (
    PurgingEquilibriumRunner,
    _purge_touches,
    _safe_rmtree,
    _trim_failed_work_dir,
)
from lpopt.vendor.masterrl.dataset import CaseData
from lpopt.vendor.masterrl.domain import CaseKey
from lpopt.vendor.masterrl.master import MasterRunError, MasterRunner

# A minimal restart-read reload deck: one %JOB_TYP restart, %JOB_IDE cycle 1,
# a %LPD_SHF body the runner overwrites with the pattern.  References the base
# restart basename so MasterRunner._assets accepts cycle 1.
_TEMPLATE_DECK = (
    "%JOB_TYP\n"
    "        1       stead                               # irrst=1 (restart)\n"
    "        MAS_RST.SEED.00\n"
    "        xsl     MAS_XSL\n"
    "        hff     MAS_HFF\n"
    "        out     MAS_OUT\n"
    "        sum     MAS_SUM\n"
    "%JOB_IDE\n"
    "        APRQ    1\n"
    "%GEN_DIM\n"
    "        10      10      27      83      85          # nx, ny, nz, nbatch, ncomp\n"
    "%LPD_SHF\n"
    "        F K1  0,\n"
    "%EXE_DEP                                            # BOC\n"
    "        0.0\n"
    "%EDT_OPT                                            # EOC restart write\n"
    "        1\n"
    "%END\n"
)

# A fixed, converging MAS_SUM: identical every cycle -> zero five-FOM deltas ->
# two consecutive matches converge the chain.  EDIT 2 / EDIT 3 share one EFPD grid.
_MAS_SUM = (
    " SUMMARY EDIT 2 : REACTIVITY\n"
    " NO. DAY EFPD PPM ERRFLX\n"
    " 1 0 0.0 800 1e-7\n"
    " 2 5 10.0 750 1e-7\n"
    " 3 10 20.0 700 1e-7\n"
    "\n"
    " SUMMARY EDIT 3 : PEAKING\n"
    " NO. DAY EFPD AO FQP FRP\n"
    " 1 0 0.0 -0.10 1.50 1.40\n"
    " 2 5 10.0 0.00 1.55 1.42\n"
    " 3 10 20.0 0.05 1.52 1.41\n"
)

# Fake MASTER: parse the %JOB_IDE cycle, write a per-cycle MAS_RST so the vendor
# chain finds exactly one *new* restart, and emit the fixed MAS_SUM.  ``FAIL_AT``
# (env) makes the run of that cycle drop a NONFINITE sentinel and exit non-zero,
# simulating a divergent chain that must leave a small diagnostic tail.
_FAKE_MASTER = r'''
import os, re, sys
from pathlib import Path

deck = Path("MAS_INP").read_text(errors="replace")
cycle = 1
lines = deck.splitlines()
for i, ln in enumerate(lines):
    if ln.strip().upper().startswith("%JOB_IDE"):
        for j in range(i + 1, len(lines)):
            s = lines[j].strip()
            if not s or s.startswith("#") or s == "/":
                continue
            ints = re.findall(r"-?\d+", s)
            if ints:
                cycle = int(ints[-1])
            break
        break

Path("MAS_OUT").write_text("BURNUP 0.0 EFPD 0.0\nBURNUP 20.0 EFPD 20.0 done\n")

fail_at = int(os.environ.get("FAKE_FAIL_AT", "0"))
if fail_at and cycle >= fail_at:
    Path("NONFINITE_FLUX").write_text("non_finite_flux\n", encoding="utf-8")
    Path("MAS_OUT").write_text("MGOUTER 1 1 NaN NaN NaN\n" * 20)
    sys.exit(1)

Path("MAS_SUM").write_text(SUMMARY_PLACEHOLDER)
Path("MAS_RST.CY%02d" % cycle).write_bytes(b"restart-cy%02d-" % cycle + b"X" * 4096)
'''


def _make_pkg(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A minimal MASTER package + base restart + template deck + fake exe."""

    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "MAS_XSL").write_bytes(b"xsl" * 4096)
    (pkg / "lib" / "MAS_HFF").write_bytes(b"hff" * 4096)

    base_restart = tmp_path / "bases" / "MAS_RST.SEED.00"
    base_restart.parent.mkdir(parents=True)
    base_restart.write_bytes(b"seed" * 4096)

    tpl_dir = tmp_path / "tpl"
    tpl_dir.mkdir()
    template = tpl_dir / "MAS_INP_cy01.inp"
    template.write_text(_TEMPLATE_DECK, encoding="utf-8")

    fake = tmp_path / "fake_master.py"
    fake.write_text(
        "SUMMARY_PLACEHOLDER = " + repr(_MAS_SUM) + "\n" + _FAKE_MASTER,
        encoding="utf-8",
    )
    return pkg, base_restart, template, fake


def _runner(
    pkg: Path,
    fake: Path,
    work_root: Path,
    *,
    keep_success: bool,
    purge_intermediate: bool,
) -> PurgingEquilibriumRunner:
    master = MasterRunner(
        pkg,
        [sys.executable, str(fake)],
        work_root=work_root,
        timeout=60.0,
        keep_success=keep_success,
    )
    return PurgingEquilibriumRunner(
        master,
        max_cycles=8,
        consecutive=2,
        keep_success=keep_success,
        purge_intermediate=purge_intermediate,
    )


def _case(template: Path, base_restart: Path) -> tuple[CaseData, object]:
    key = CaseKey("K1_K2", 121)
    pattern = random_genome(random.Random(3), "K1_K2", 30).to_pattern()
    assert pattern.feed == 121
    case = CaseData(
        key=key, cell=0.0, records=(),
        template_path=template, restart_path=base_restart,
    )
    return case, pattern


def _cycle_dirs(work_root: Path) -> list[Path]:
    return [p for p in work_root.iterdir() if p.is_dir()] if work_root.exists() else []


# --------------------------------------------------------------------------- #
# 1) converged + kept: ONLY the final equilibrium cycle survives
# --------------------------------------------------------------------------- #
def test_purge_keeps_only_final_equilibrium_cycle(tmp_path: Path) -> None:
    pkg, base_restart, template, fake = _make_pkg(tmp_path)
    work_root = tmp_path / "work"
    runner = _runner(pkg, fake, work_root, keep_success=True, purge_intermediate=True)
    case, pattern = _case(template, base_restart)

    eq = runner.run(case, pattern)

    assert eq.converged is True
    assert eq.n_cycles >= 3, "identical metrics should converge after >=3 cycles"

    survivors = _cycle_dirs(work_root)
    assert len(survivors) == 1, f"only the final cycle dir may survive, got {survivors}"
    final = survivors[0]
    # The survivor is the final equilibrium cycle: MAS_SUM + its final restart.
    assert (final / "MAS_SUM").is_file()
    final_restarts = list(final.glob("MAS_RST.*"))
    assert final_restarts, "final cycle must retain a MAS_RST.* for promotion/harvest"
    # The runner's reported final restart lives in this surviving dir.
    assert eq.cycles[-1].restart_path.is_file()
    assert eq.cycles[-1].restart_path.parent == final
    # retained_work_dirs reflects the on-disk truth (only the final).
    assert list(eq.retained_work_dirs) == [final]


# --------------------------------------------------------------------------- #
# 2) the surviving final restart is promotable (CaseAssetResolver promotion path)
# --------------------------------------------------------------------------- #
def test_final_restart_is_promotable(tmp_path: Path) -> None:
    pkg, base_restart, template, fake = _make_pkg(tmp_path)
    work_root = tmp_path / "work"
    runner = _runner(pkg, fake, work_root, keep_success=True, purge_intermediate=True)
    case, pattern = _case(template, base_restart)

    eq = runner.run(case, pattern)
    final_restart = eq.cycles[-1].restart_path
    assert final_restart.is_file()

    resolver = CaseAssetResolver(pkg, promoted_root=tmp_path / "promoted")
    dest = resolver.promote(CaseKey("K1_K2", 121), final_restart)
    assert dest.is_file()
    assert dest.read_bytes() == final_restart.read_bytes()


# --------------------------------------------------------------------------- #
# 3) converged + NOT kept: vendor cleans the final too -> nothing survives
# --------------------------------------------------------------------------- #
def test_purge_not_kept_leaves_nothing(tmp_path: Path) -> None:
    pkg, base_restart, template, fake = _make_pkg(tmp_path)
    work_root = tmp_path / "work"
    runner = _runner(pkg, fake, work_root, keep_success=False, purge_intermediate=True)
    case, pattern = _case(template, base_restart)

    eq = runner.run(case, pattern)
    assert eq.converged is True
    assert _cycle_dirs(work_root) == []
    assert list(eq.retained_work_dirs) == []


# --------------------------------------------------------------------------- #
# 4) purge_intermediate=False restores the vendor lifecycle (all cycles kept)
# --------------------------------------------------------------------------- #
def test_purge_disabled_keeps_all_cycles(tmp_path: Path) -> None:
    pkg, base_restart, template, fake = _make_pkg(tmp_path)
    work_root = tmp_path / "work"
    runner = _runner(pkg, fake, work_root, keep_success=True, purge_intermediate=False)
    case, pattern = _case(template, base_restart)

    eq = runner.run(case, pattern)
    survivors = _cycle_dirs(work_root)
    assert len(survivors) == eq.n_cycles >= 3, (
        "with purge disabled every retained cycle dir must remain (vendor behavior)"
    )


# --------------------------------------------------------------------------- #
# 5) failed chain: only the failing cycle's dir survives, trimmed to a small tail
# --------------------------------------------------------------------------- #
def test_failed_chain_keeps_trimmed_diagnostic_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg, base_restart, template, fake = _make_pkg(tmp_path)
    work_root = tmp_path / "work"
    # keep_success=False is the production default; the failing cycle dir is still
    # retained by the vendor MasterRunError, and prior successful cycles are gone.
    runner = _runner(pkg, fake, work_root, keep_success=False, purge_intermediate=True)
    case, pattern = _case(template, base_restart)

    # Diverge on the 3rd cycle (after two intermediate successes).
    monkeypatch.setenv("FAKE_FAIL_AT", "3")
    with pytest.raises(MasterRunError):
        runner.run(case, pattern)

    survivors = _cycle_dirs(work_root)
    assert len(survivors) == 1, f"only the failing cycle dir may survive, got {survivors}"
    failed = survivors[0]
    # Diagnostic tail preserved: the NONFINITE sentinel + MAS_OUT the classifier reads.
    assert (failed / "NONFINITE_FLUX").is_file()
    assert (failed / "MAS_OUT").is_file()
    # Trimmed: the big staged libraries + every restart are gone (kept small).
    assert not (failed / "MAS_XSL").exists()
    assert not (failed / "MAS_HFF").exists()
    assert not list(failed.glob("MAS_RST.*"))


# --------------------------------------------------------------------------- #
# 6) _trim_failed_work_dir is a pure, defensive helper
# --------------------------------------------------------------------------- #
def test_trim_failed_work_dir_keeps_only_small_tail(tmp_path: Path) -> None:
    d = tmp_path / "wd"
    d.mkdir()
    for name in ("MAS_OUT", "MAS_INP", "NONFINITE_FLUX", "MASTER.stdout"):
        (d / name).write_text("x")
    for name in ("MAS_XSL", "MAS_HFF", "MAS_RST.CY03", "MAS_SUM", "MAS_PPI.01"):
        (d / name).write_bytes(b"big" * 1000)

    _trim_failed_work_dir(d)

    assert (d / "MAS_OUT").exists() and (d / "NONFINITE_FLUX").exists()
    assert (d / "MAS_INP").exists() and (d / "MASTER.stdout").exists()
    assert not (d / "MAS_XSL").exists()
    assert not (d / "MAS_HFF").exists()
    assert not (d / "MAS_RST.CY03").exists()
    assert not (d / "MAS_SUM").exists()
    assert not (d / "MAS_PPI.01").exists()
    # Defensive: a None / missing path must not raise.
    _trim_failed_work_dir(None)
    _trim_failed_work_dir(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# 7) config knobs: [produce] purge_intermediate default True, consecutive default 2
# --------------------------------------------------------------------------- #
def test_produce_config_purge_and_consecutive_defaults() -> None:
    cfg = ProduceConfig()
    assert cfg.purge_intermediate is True
    assert cfg.consecutive == 2


# --------------------------------------------------------------------------- #
# 8) bug B fence: the intermediate-cycle purge can NEVER touch the staged
#    cases_dir tree (a purge target that resolves onto a case_dir — the thing
#    master.py:_assets requires to exist — is refused, even when it would pass
#    the work_root fence).  Defense-in-depth for the 56-way round-2.
# --------------------------------------------------------------------------- #
def test_purge_fence_never_touches_cases_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cases = run_dir / "produce_cases"
    case_dir = cases / "K1_K2" / "feed121"   # what _assets requires to exist
    case_dir.mkdir(parents=True)
    (case_dir / "MAS_INP_cy12.inp").write_text("deck", encoding="utf-8")

    # _purge_touches: every overlap on the cases line is a conflict; disjoint is not.
    assert _purge_touches(cases, cases)          # equal
    assert _purge_touches(case_dir, cases)       # descendant (a staged case dir)
    assert _purge_touches(run_dir, cases)        # ancestor (rmtree would take cases)
    work = tmp_path / "work" / "worker_00" / "cyc_ab"
    work.mkdir(parents=True)
    (work / "MAS_SUM").write_bytes(b"x" * 512)
    assert not _purge_touches(work, cases)       # disjoint work dir — allowed
    assert not _purge_touches(None, cases) and not _purge_touches(work, None)

    # _safe_rmtree(protect=cases) refuses the case dir and any ancestor of it,
    # but still purges a disjoint work dir under its work_root.
    _safe_rmtree(case_dir, protect=cases)
    assert case_dir.exists(), "purge must NOT delete a staged case_dir"
    _safe_rmtree(run_dir, protect=cases)         # ancestor delete would remove cases
    assert case_dir.exists(), "purge must NOT delete an ancestor of cases_dir"
    _safe_rmtree(work, work_root=tmp_path / "work", protect=cases)
    assert not work.exists(), "a disjoint per-cycle work dir is still purged"

    # _trim_failed_work_dir(protect=cases) refuses to empty a dir inside the tree.
    (case_dir / "MAS_XSL").write_bytes(b"big" * 1000)
    _trim_failed_work_dir(case_dir, protect=cases)
    assert (case_dir / "MAS_XSL").exists()
    assert (case_dir / "MAS_INP_cy12.inp").exists()

    # wiring: PurgingEquilibriumRunner records the protected cases_dir (default None).
    master = MasterRunner(tmp_path / "pkg", ["true"], work_root=tmp_path / "wr")
    assert PurgingEquilibriumRunner(master, protect_dir=cases)._protect_dir == cases
    assert PurgingEquilibriumRunner(master)._protect_dir is None
