"""Bug A regression: the user_criteria CAMPAIGN verification path (a real-MASTER
``WaveVerifier`` with an injected purge, exactly as ``UserCriteriaDriver`` builds
it) must NEVER surface ``PermissionError: [Errno 13] Permission denied: '.'``.

Root cause (see ``verify.py``): the free-search reaches a pair with no package
template deck, so ``ResolvedAssets.template_deck_path`` is ``None``; the old
``_build_case_data`` substituted a ``Path('.')`` sentinel that the vendor
``EquilibriumRunner`` then ``read_bytes()`` (and ``_link_or_copy`` ``copy2()``) —
reading the CWD directory on Windows raises ``PermissionError('.')``.

This module drives the SAME construction as the driver (real default factory +
fake exe, ``purge_intermediate=True``, ``stage_decks`` on) and asserts:

* a properly-resolved wave CONVERGES with no PermissionError, and the per-cycle
  purge still removes intermediate cycles (only the final equilibrium dir left);
* a missing-template entry becomes a clean, descriptive ``MissingCaseAssetError``
  ``error`` outcome — never a PermissionError — and the process CWD is untouched;
* the purge guard helpers refuse to delete ``.``/CWD/any non-subpath of the
  worker ``work_root`` (log-and-skip, never raise, never touch the target).
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

from lpopt.search.assets import CaseAssetResolver, ResolvedAssets
from lpopt.search.genome import random_genome
from lpopt.search.verify import (
    MissingCaseAssetError,
    WaveEntry,
    WaveVerifier,
    _is_strict_subpath,
    _safe_rmtree,
    _trim_failed_work_dir,
)
from lpopt.vendor.masterrl.domain import CaseKey

# A valid reload deck (cycle 12) referencing the base restart basename; passes
# ``validate_reload_deck`` (one %JOB_TYP restart, %EXE_DEP BOC, %EDT_OPT EOC
# write, %GEN_DIM matching the default library dims).
_RELOAD_DECK = (
    "%JOB_TYP\n"
    "        1       stead                               # irrst=1 (restart)\n"
    "        MAS_RST.SEED.01\n"
    "        xsl     MAS_XSL\n"
    "        hff     MAS_HFF\n"
    "        out     MAS_OUT\n"
    "        sum     MAS_SUM\n"
    "%JOB_IDE\n"
    "        APRQ    12\n"
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

# Fake MASTER: parse the %JOB_IDE cycle, emit the fixed (converging) MAS_SUM and a
# per-cycle MAS_RST so the vendor chain finds exactly one new restart each cycle.
_FAKE_MASTER = r'''
import re
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
Path("MAS_SUM").write_text(SUMMARY_PLACEHOLDER)
Path("MAS_RST.CY%02d" % cycle).write_bytes(b"restart-cy%02d-" % cycle + b"X" * 4096)
'''


def _make_pkg(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "MAS_XSL").write_bytes(b"xsl" * 4096)
    (pkg / "lib" / "MAS_HFF").write_bytes(b"hff" * 4096)

    restart = tmp_path / "restarts" / "MAS_RST.SEED.01"
    restart.parent.mkdir(parents=True)
    restart.write_bytes(b"seed" * 4096)

    template = tmp_path / "tpl" / "MAS_INP_cy12.inp"
    template.parent.mkdir(parents=True)
    template.write_text(_RELOAD_DECK, encoding="utf-8")

    fake = tmp_path / "fake_master.py"
    fake.write_text(
        "SUMMARY_PLACEHOLDER = " + repr(_MAS_SUM) + "\n" + _FAKE_MASTER,
        encoding="utf-8",
    )
    return pkg, restart, template, fake


def _campaign_verifier(
    tmp_path: Path, pkg: Path, fake: Path, *, keep_success: bool = True
) -> WaveVerifier:
    """A WaveVerifier built EXACTLY like UserCriteriaDriver's real-MASTER path:
    ``run_dir/master`` + real default factory (fake exe) + purge on."""

    return WaveVerifier(
        run_dir=tmp_path / "run" / "master",
        package_root=pkg,
        executable=[sys.executable, str(fake)],
        workers=2,
        max_cycles=16,
        consecutive=2,
        keep_success=keep_success,
        resolver=CaseAssetResolver(pkg),
        purge_intermediate=True,
    )


def _entries(restart: Path, template: Path, *, count: int) -> list[WaveEntry]:
    rng = random.Random(3)
    seen: dict[str, object] = {}
    while len(seen) < count:
        p = random_genome(rng, "K1_K2", 29).to_pattern()  # feed 117
        seen[p.digest] = p
    ck = CaseKey("K1_K2", 117)
    resolved = ResolvedAssets(
        case_key=ck,
        restart_path=restart,
        template_deck_path=template,
        fallback_level=3,
        restart_provenance="pair_ecore:MAS_RST.SEED.01",
    )
    return [
        WaveEntry(pat, ck, resolved, {"pair": "K1_K2", "phase": "lean_r1"})
        for pat in seen.values()
    ]


def _worker_cycle_dirs(verifier: WaveVerifier) -> list[Path]:
    root = verifier.work_root
    if not root.exists():
        return []
    return [
        p
        for worker in root.iterdir()
        if worker.is_dir()
        for p in worker.iterdir()
        if p.is_dir()
    ]


# --------------------------------------------------------------------------- #
# 1) resolved wave CONVERGES; no PermissionError; intermediates purged
# --------------------------------------------------------------------------- #
def test_campaign_wave_converges_and_purges(tmp_path: Path) -> None:
    pkg, restart, template, fake = _make_pkg(tmp_path)
    verifier = _campaign_verifier(tmp_path, pkg, fake, keep_success=True)
    entries = _entries(restart, template, count=4)

    outcomes = verifier.evaluate_wave(entries)

    assert len(outcomes) == 4
    assert all(o.status == "converged" for o in outcomes), [
        (o.status, o.failure) for o in outcomes
    ]
    # The exact Bug A signature must never appear.
    assert all("Permission denied" not in (o.failure or "") for o in outcomes)
    assert all(o.n_cycles >= 3 for o in outcomes)

    # Per-cycle purge held: with keep_success the ONLY survivors are the final
    # equilibrium cycle of each chain (one per entry), not every intermediate.
    survivors = _worker_cycle_dirs(verifier)
    assert survivors, "final equilibrium cycle dirs must remain (keep_success)"
    for final in survivors:
        assert (final / "MAS_SUM").is_file()
        assert list(final.glob("MAS_RST.*")), "final dir keeps a restart"
    # 4 entries -> at most 4 survivors (one final each); intermediates are gone.
    assert len(survivors) <= 4


# --------------------------------------------------------------------------- #
# 2) missing template -> clean MissingCaseAssetError, never PermissionError('.')
# --------------------------------------------------------------------------- #
def test_campaign_missing_template_is_clean_error_not_permission(
    tmp_path: Path,
) -> None:
    pkg, restart, _template, fake = _make_pkg(tmp_path)
    verifier = _campaign_verifier(tmp_path, pkg, fake, keep_success=False)

    ck = CaseKey("J2_L3", 121)  # an exotic free-search pair with no package deck
    pat = random_genome(random.Random(7), "J2_L3", 30).to_pattern()
    resolved = ResolvedAssets(
        case_key=ck,
        restart_path=restart,          # a restart resolved, but NO template deck
        template_deck_path=None,
        fallback_level=3,
        restart_provenance="pair_ecore:MAS_RST.SEED.01",
    )
    entry = WaveEntry(pat, ck, resolved, {"pair": "J2_L3", "phase": "lean_r1"})

    cwd_before = os.getcwd()
    (outcome,) = verifier.evaluate_wave([entry])

    assert outcome.status == "error"
    assert "Permission denied" not in outcome.failure
    assert "MissingCaseAssetError" in outcome.failure
    assert "J2_L3" in outcome.failure
    # The process CWD (and its contents) are entirely untouched.
    assert os.getcwd() == cwd_before
    assert Path(cwd_before).is_dir()


# --------------------------------------------------------------------------- #
# 3) purge guard: never delete '.'/CWD/a non-subpath of the worker work_root
# --------------------------------------------------------------------------- #
def test_safe_rmtree_refuses_paths_outside_work_root(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    inside = work_root / "cycle_01"
    inside.mkdir()
    (inside / "f").write_text("x")

    # A strict subpath IS deleted.
    _safe_rmtree(inside, work_root=work_root)
    assert not inside.exists()

    # A path OUTSIDE the work_root is refused (log-and-skip), left intact.
    outside = tmp_path / "keepme"
    outside.mkdir()
    (outside / "f").write_text("x")
    _safe_rmtree(outside, work_root=work_root)
    assert outside.exists() and (outside / "f").exists()

    # work_root itself (== root, not a strict subpath) is refused.
    _safe_rmtree(work_root, work_root=work_root)
    assert work_root.exists()

    # A Path('.') / CWD is refused both with and without a declared root.
    _safe_rmtree(Path("."), work_root=work_root)
    _safe_rmtree(Path("."))
    assert Path(os.getcwd()).is_dir()


def test_is_strict_subpath_semantics(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    assert _is_strict_subpath(root / "a", root) is True
    assert _is_strict_subpath(root / "a" / "b", root) is True
    assert _is_strict_subpath(root, root) is False           # equal, not strict
    assert _is_strict_subpath(tmp_path, root) is False        # parent, outside
    assert _is_strict_subpath(Path("."), root) is False
    assert _is_strict_subpath(None, root) is False
    assert _is_strict_subpath(root / "a", None) is False


def test_trim_failed_work_dir_refuses_outside_root(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    # A directory OUTSIDE the work_root must be left fully intact by a guarded trim.
    outside = tmp_path / "not_a_worker_dir"
    outside.mkdir()
    (outside / "MAS_XSL").write_bytes(b"big")
    (outside / "MAS_RST.CY03").write_bytes(b"big")

    _trim_failed_work_dir(outside, work_root=work_root)

    assert (outside / "MAS_XSL").exists()
    assert (outside / "MAS_RST.CY03").exists()

    # A strict subpath IS trimmed (big files gone, small tail kept).
    inside = work_root / "cycle_03"
    inside.mkdir()
    (inside / "MAS_OUT").write_text("tail")
    (inside / "MAS_XSL").write_bytes(b"big")
    _trim_failed_work_dir(inside, work_root=work_root)
    assert (inside / "MAS_OUT").exists()
    assert not (inside / "MAS_XSL").exists()
