"""Ship the v3 corpus to 238, launch ``lpopt.policy.train_v3`` there, poll, pull.

    python train_policy_v3.py --ts policy_v3 --seeds 5 --epochs 120 \
        --extra "--holdout-cell N1_N2/f113/ga80 --base-seed 20260831 --protocol revB"

Identical in shape to ``train_policy_v2.py`` — same ``lpopt.remote`` push/status/
pull, same heartbeat/rc/DONE markers, same **GPU-1 pinning via ``lpopt_gpu1.inp``
so GPU 0 is not touched**.  Three files go over by ``ship_file`` because ``push``
does not carry ``data/policy`` or ``data/design``: the v3 corpus, the fuel table
(the Gd/lattice descriptors read it) and the BLIND policy-v2 baseline
predictions.

The command this builds is the one registered in
``data/reports/policy_v3_prereg_20260831.md`` §8b, verbatim, with ONE declared
difference: the corpus is ``data/policy/steps_v3.parquet``, not a re-mined
``data/policy/steps.parquet``.  §8-A's in-place re-mine would change the bytes
the shipped ``data/models/policy_v2`` stamps as its ``corpus_sha256``, and those
bytes are what ``tests/test_policy_prior.py`` re-reads to prove v2 serving still
reproduces v2 training — i.e. it would invalidate the very baseline §5c requires.
A new file gets the true provenance (``lpopt.policy.v3.provenance_v3``) with the
v2 artefacts intact.  Nothing else in §8b moves.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lpopt.remote import (
    RemoteSettings, pick_gpu, pull, push, run_ssh, ship_file, status,
)

REPO = Path(__file__).resolve().parent
#: The v3 corpus (85 -> 107 columns, ``policy_v2_corpus.py backfill-v3``).
STEPS = "data/policy/steps_v3.parquet"
FUEL_TYPES = "data/store/fuel_types.parquet"
#: Emitted and hashed BEFORE any v3 weight existed (prereg §5c / §8-A).
V2_BASELINE = "data/design/policy_v3_v2_baseline.csv"
#: The registered held-out cell.  Not a default the caller may drift: it is in
#: the launched command so ``runs/<ts>/run.sh`` records what was actually held
#: out, and ``--extra`` can only repeat it.
HOLDOUT_CELL = "N1_N2/f113/ga80"

RUN_SCRIPT = """#!/bin/bash
set -o pipefail
RUN="$HOME/{workdir}/runs/{ts}"
mkdir -p "$RUN"
cd "$HOME/{workdir}"
( while true; do date +%s > "$RUN/heartbeat"; sleep 15; done ) &
HB=$!
CUDA_VISIBLE_DEVICES={gpu} {python} -m lpopt.policy.train_v3 {args} \
    --out-dir "runs/{ts}" > "$RUN/train.log" 2>&1
RC=$?
kill $HB 2>/dev/null
echo $RC > "$RUN/rc"
if [ $RC -eq 0 ]; then touch "$RUN/DONE"; else touch "$RUN/FAILED"; fi
"""


#: The v3.1 round (``policy_v31_prereg_20260831_DRAFT.md`` §9c).  A DIFFERENT
#: corpus, a different feature cache, a different held-out cell and a different
#: blind v2 baseline — none of the v3 artefacts above is re-used or overwritten.
#: Only the SPLIT EMISSION step of the round is launchable today; the training
#: step is blocked on prereg delta D (see :func:`train_args`).
STEPS_V31 = "data/policy/steps_v31.parquet"
V2_BASELINE_V31 = "data/design/policy_v31_v2_baseline.csv"
HOLDOUT_CELL_V31 = "E1_E2/f109/ga80"
#: The FROZEN STEP 0-a assignment (freeze stamp §S0.5).  Passing it is what turns
#: ``--xfit-k`` from the split EMISSION into the consumption of a hashed
#: assignment, and it is what lifts ``assert_v3_path_untouched``'s refusal.
SPLITS_V31 = "data/policy/v31_split/splits_v31.csv"
#: The v3.1 round's REGISTERED launcher values (prereg §9b / §9c and the STEP 0-b
#: stamp §S0b.10).  ``--v31`` seeds the parser with these instead of the v3
#: round's, so ``--v31 --print-command`` renders the command that was registered
#: rather than the v3 defaults wearing the v3.1 flags; passing the flag
#: explicitly still wins, and the registered launch line
#: ``--v31 --ts policy_v31 --seeds 3 --base-seed 20260903`` is unchanged by it.
V31_DEFAULTS = {"ts": "policy_v31", "seeds": 3, "base_seed": 20260903}
#: Arm ii writes its own run directory.  ``runs/policy_v31`` is arm i's and is
#: never re-entered: the two arms are compared, so neither may overwrite the
#: other's ``metrics.json``.
ARM_II_TS = "policy_v31_arm_ii"


def _round(args) -> dict[str, str]:
    """The four artefacts that differ between the v3 and v3.1 rounds."""
    if getattr(args, "v31", False):
        return {"steps": STEPS_V31, "cache": "data/policy/_feature_cache_v31.npz",
                "baseline": V2_BASELINE_V31, "holdout": HOLDOUT_CELL_V31}
    return {"steps": STEPS, "cache": "data/policy/_feature_cache_v3.npz",
            "baseline": V2_BASELINE, "holdout": HOLDOUT_CELL}


def train_args(args) -> str:
    """The exact ``lpopt.policy.train_v3`` argument string of prereg §8b.

    Every §8a knob is written out here rather than left to the module defaults,
    so ``runs/<ts>/run.sh`` on the box is a complete record of the protocol and a
    later default change cannot silently re-define a finished run.

    ``--v31`` renders the §9c training command now that **prereg delta D**
    (``lpopt/policy/metrics_v31.py``) has landed.  Three flags carry the whole
    difference and each is written out rather than defaulted:

    * ``--stage2 on`` — the frozen-trunk ``fxy`` branch of §2b;
    * ``--xfit-k 5 --splits data/policy/v31_split/splits_v31.csv`` — the
      component-blocked cross-fit CONSUMED from the hashed STEP 0-a emission.
      Without ``--splits`` the same flag is still the emission step and
      ``assert_v3_path_untouched`` refuses the pair, which is why the launcher
      never renders one without the other;
    * ``--lam-grid 0,0.3,1.0`` — the registered grid.  It is **not** an arm
      (§9d): all three are trained and ONE is selected on ``val`` mean Spearman
      alone (§2c), and only the selected one reaches the gate.  The
      pre-registered expectation is that 0 wins.

    ``--stage2 on`` WITHOUT ``--xfit-k`` stays refused for the life of the round
    and is never rendered: it would train on v3's single alternating split (37
    parents with >= 8 F_xy candidates, 16 live -- the row §3a marks REJECTED)
    while stamping the checkpoint v3.1.  Without ``--v31`` not one character of
    the v3 command moves.
    """
    r = _round(args)
    if getattr(args, "v31", False):
        return (
            f"--steps {r['steps']} --fuel-types {FUEL_TYPES} "
            f"--cache {r['cache']} "
            f"--v2-baseline {r['baseline']} "
            f"--holdout-cell \"{r['holdout']}\" "
            f"--xfit-k {args.xfit_k} --splits {SPLITS_V31} "
            f"--stage2 on --lam-grid {args.lam_grid} "
            f"--teacher raw --teacher-temp {args.teacher_temp} "
            f"--teacher-eps {args.teacher_eps} "
            f"--stage2-lr {args.stage2_lr} "
            f"--stage2-epochs {args.stage2_epochs} "
            f"--seeds {args.seeds} --base-seed {args.base_seed} "
            f"--epochs {args.epochs} --patience {args.patience} "
            f"--batch-size {args.batch_size} --lr {args.lr} "
            f"--weight-decay {args.weight_decay} "
            f"--width {args.width} --n-blocks {args.n_blocks} "
            f"--protocol {args.protocol} "
            f"--device auto --num-workers {args.num_workers} {args.extra}"
        ).strip()
    extra = ""
    return (
        f"--steps {r['steps']} --fuel-types {FUEL_TYPES} "
        f"--cache {r['cache']} "
        f"--v2-baseline {r['baseline']} "
        f"--holdout-cell \"{r['holdout']}\" "
        f"--seeds {args.seeds} --base-seed {args.base_seed} "
        f"--epochs {args.epochs} --patience {args.patience} "
        f"--batch-size {args.batch_size} --lr {args.lr} "
        f"--weight-decay {args.weight_decay} "
        f"--width {args.width} --n-blocks {args.n_blocks} "
        f"--protocol {args.protocol} "
        f"--device auto --num-workers {args.num_workers}{extra} {args.extra}"
    ).strip()


def arm_ii_args(args) -> str:
    """Prereg §9d **arm ii** — the control, now that ``--no-burnt`` exists.

    §9d registers it as ``--no-burnt --lam 0 --stage2 off``: the same corpus and
    the same held-out cell as arm i, refit as v3 with the two burnt columns
    dropped, so "cross-fit / feature / listwise" becomes a decomposition instead
    of a post-hoc guess.  Three notes on what is and is not rendered:

    * ``--no-burnt`` is what makes this a control rather than the v3 round under
      another name: the corpus is ``steps_v31.parquet``, and ``featurize_round``
      REQUIRES the burnt columns to be present before it drops them and stamps
      ``burnt: off``.
    * ``--lam-grid 0`` is written out, not defaulted, because the zero is the
      statement.  ``train_v3`` admits exactly this one value with ``--stage2
      off`` and refuses any other.
    * ``--xfit-k``/``--splits`` are NOT rendered.  With ``--stage2 off`` the
      cross-fit flag is still the split EMISSION step, which returns before a
      weight is trained; rendering it would produce a run that touches ``DONE``
      with no model.  So arm ii is the FEATURE half of §9d's decomposition on
      v3's single alternating split, and the cross-fit half stays open until the
      out-of-fold loop is reachable with stage 2 off.  That limitation is
      printed rather than papered over, which is the same discipline that kept
      this arm unrendered while ``--no-burnt`` did not exist.

    The out-dir is :data:`ARM_II_TS`, never arm i's ``runs/policy_v31``.
    """
    r = _round(args)
    return (
        f"--steps {STEPS_V31} --fuel-types {FUEL_TYPES} "
        f"--cache {r['cache']} "
        f"--v2-baseline {V2_BASELINE_V31} "
        f"--holdout-cell \"{HOLDOUT_CELL_V31}\" "
        f"--no-burnt --lam-grid 0 --stage2 off "
        f"--seeds {args.seeds} --base-seed {args.base_seed} "
        f"--epochs {args.epochs} --patience {args.patience} "
        f"--batch-size {args.batch_size} --lr {args.lr} "
        f"--weight-decay {args.weight_decay} "
        f"--width {args.width} --n-blocks {args.n_blocks} "
        f"--protocol {args.protocol} "
        f"--device auto --num-workers {args.num_workers}"
    ).strip()


def xfit_command(args) -> str:
    """The §3a split-emission step, which runs BEFORE the training command.

    It writes ``runs/<ts>/splits_v31.csv`` and ``xfit_census.json`` and trains
    nothing, so it is safe to run and hash while no v3.1 weight exists — which
    is the same order discipline the blind v2 re-emission of §3b needs.
    """
    r = _round(args)
    return (f"-m lpopt.policy.train_v3 --steps {r['steps']} "
            f"--fuel-types {FUEL_TYPES} --cache {r['cache']} "
            f'--holdout-cell "{r["holdout"]}" '
            f"--xfit-k {args.xfit_k} --base-seed {args.base_seed} "
            f"--out-dir runs/{args.ts}")


def occupancy(s: RemoteSettings) -> str:
    cp = run_ssh(s, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
                    "utilization.gpu --format=csv,noheader; echo '--'; "
                    "tmux ls 2>/dev/null || echo 'no tmux'", timeout=60)
    return cp.stdout.decode(errors="replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="lpopt_gpu1.inp")
    ap.add_argument("--ts", default="policy_v3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=20260831)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--weight-decay", default="1e-4")
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--protocol", default="revB")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--extra", default="")
    ap.add_argument("--print-command", action="store_true",
                    help="render the remote command and exit; touches nothing")
    # ---- v3.1 round.  Off by default: without --v31 this launcher ships and
    # launches exactly what it shipped and launched for v3. ------------------ #
    ap.add_argument("--v31", action="store_true",
                    help="the policy v3.1 round: the 111-column corpus, the "
                         "E1_E2/f109/ga80 holdout, cross-fit and stage 2")
    ap.add_argument("--xfit-k", type=int, default=5)
    ap.add_argument("--lam-grid", default="0,0.3,1.0")
    ap.add_argument("--teacher-temp", default="0.060")
    ap.add_argument("--teacher-eps", default="0.10")
    ap.add_argument("--stage2-lr", default="1e-4")
    ap.add_argument("--stage2-epochs", type=int, default=40)
    args = ap.parse_args(argv)
    if args.v31:
        # argparse only fills a default for a dest the namespace does not
        # already carry, so re-parsing onto a namespace pre-seeded with
        # V31_DEFAULTS swaps the v3 round's defaults for the REGISTERED v3.1
        # ones while an explicitly passed flag still wins.  Without this
        # `--v31 --print-command` printed `--seeds 5 --base-seed 20260831
        # --out-dir runs/policy_v3` -- the v3 round's numbers under the v3.1
        # flags, i.e. a command nobody registered.
        args = ap.parse_args(argv, namespace=argparse.Namespace(**V31_DEFAULTS))

    if args.print_command:
        if args.v31:
            print(f"# 1) split emission -- ALREADY DONE and frozen; re-run only "
                  f"to reproduce {SPLITS_V31}\n"
                  f"<venv>/bin/python {xfit_command(args)}\n"
                  f"# 2) training (arm i, v3.1): lambda grid 0,0.3,1.0 is "
                  f"SELECTED on val, not swept as arms (§2c/§9d)")
        print(f"CUDA_VISIBLE_DEVICES=<gpu> <venv>/bin/python -m "
              f"lpopt.policy.train_v3 {train_args(args)} "
              f"--out-dir runs/{args.ts}")
        if args.v31:
            print("# 3) training (arm ii, the §9d control: v3 refit on the SAME "
                  "corpus with the two burnt columns dropped, 53 -> 51).  "
                  "`--no-burnt` now exists, so the arm is rendered; "
                  "featurize_round REQUIRES the v3.1 corpus before it drops "
                  "them and the run stamps `burnt: off`.  `--xfit-k`/`--splits` "
                  "are absent on purpose: with `--stage2 off` the cross-fit "
                  "flag is still the split EMISSION and would return before a "
                  "weight exists, so this arm decomposes the FEATURE half of "
                  "§9d only and the cross-fit half stays open.")
            print(f"CUDA_VISIBLE_DEVICES=<gpu> <venv>/bin/python -m "
                  f"lpopt.policy.train_v3 {arm_ii_args(args)} "
                  f"--out-dir runs/{ARM_II_TS}")
        return 0

    r = _round(args)
    ship = ((r["steps"], FUEL_TYPES, r["baseline"], SPLITS_V31) if args.v31
            else (r["steps"], FUEL_TYPES, r["baseline"]))
    for rel in ship:
        if not (REPO / rel).is_file():
            print(f"missing {rel}; the v3 corpus is built by "
                  f"`python policy_v2_corpus.py backfill-v3 --apply` and the "
                  f"blind v2 baseline by `python -m lpopt.policy.train_v3 "
                  f"--emit-v2-baseline {V2_BASELINE}`", file=sys.stderr)
            return 1

    s = RemoteSettings.from_input(str(REPO / args.input))
    print("=== occupancy ===\n" + occupancy(s), flush=True)

    if not args.no_push:
        print("=== push src ===", flush=True)
        push(s, repo_root=REPO, install=True, data=False)
    print("=== ship corpus + fuel table + blind v2 baseline ===", flush=True)
    for rel in ship:
        p = ship_file(s, REPO / rel, s.home_rel(rel))
        print(f"  {rel} -> {p}", flush=True)

    gpu = pick_gpu(s)
    workdir = s.workdir.replace("~/", "").rstrip("/")
    script = RUN_SCRIPT.format(workdir=workdir, ts=args.ts, gpu=gpu,
                               python=s.venv_python.replace("~", "$HOME"),
                               args=train_args(args))
    run_ssh(s, f"mkdir -p {s.workdir}/runs/{args.ts}", timeout=60)
    run_ssh(s, f"cat > {s.workdir}/runs/{args.ts}/run.sh",
            input_bytes=script.encode(), timeout=60)
    cp = run_ssh(s, f"tmux new-session -d -s {s.tmux_prefix}_{args.ts} "
                    f"'bash $HOME/{workdir}/runs/{args.ts}/run.sh' && echo LAUNCHED",
                 timeout=60)
    if b"LAUNCHED" not in cp.stdout:
        print(cp.stdout.decode(errors="replace"), file=sys.stderr)
        return 1
    print(f"launched ts={args.ts} gpu={gpu} session={s.tmux_prefix}_{args.ts}",
          flush=True)

    while True:
        time.sleep(args.poll)
        if status(s, args.ts)["state"] in ("done", "failed"):
            break
    print("=== pull ===", flush=True)
    pull(s, args.ts, local_root=REPO / "data" / "models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
