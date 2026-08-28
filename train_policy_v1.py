"""Ship the policy corpus to 238, launch `lpopt.policy.train` there, poll, pull.

    python train_policy_v1.py --ts policy_v1

`lpopt remote push/status/pull` are reused verbatim; only `train` is replaced,
because `lpopt/remote.py`'s run template hardcodes ``-m lpopt.model.train``.
The run.sh written here is otherwise byte-for-byte the same shape as the one
`data/models/s1e/run.sh` records — same heartbeat subshell, same rc file, same
DONE/FAILED markers — so `status` and `pull` need no change.

`push` ships `lpopt/` + `pyproject.toml` + `data/store` + `data/splits`; it does
NOT ship `data/policy`, so the step corpus goes over separately via
:func:`lpopt.remote.ship_file` (cat-over-ssh, because this repo's local path has
Korean characters, a ``+`` and parentheses and scp mangles it).
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
STEPS = "data/policy/steps.parquet"

RUN_SCRIPT = """#!/bin/bash
set -o pipefail
RUN="$HOME/{workdir}/runs/{ts}"
mkdir -p "$RUN"
cd "$HOME/{workdir}"
( while true; do date +%s > "$RUN/heartbeat"; sleep 15; done ) &
HB=$!
CUDA_VISIBLE_DEVICES={gpu} {python} -m lpopt.policy.train {args} \
    --out-dir "runs/{ts}" > "$RUN/train.log" 2>&1
RC=$?
kill $HB 2>/dev/null
echo $RC > "$RUN/rc"
if [ $RC -eq 0 ]; then touch "$RUN/DONE"; else touch "$RUN/FAILED"; fi
"""


def occupancy(s: RemoteSettings) -> str:
    cp = run_ssh(s, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
                    "utilization.gpu --format=csv,noheader; echo '--'; "
                    "tmux ls 2>/dev/null || echo 'no tmux'", timeout=60)
    return cp.stdout.decode(errors="replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="lpopt_gpu1.inp")
    ap.add_argument("--ts", default="policy_v1")
    ap.add_argument("--arms", default="cnn,mlp")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--extra", default="")
    args = ap.parse_args(argv)

    s = RemoteSettings.from_input(str(REPO / args.input))
    print("=== occupancy ===\n" + occupancy(s), flush=True)

    if not args.no_push:
        print("=== push src ===", flush=True)
        push(s, repo_root=REPO, install=True, data=False)
    print("=== ship corpus ===", flush=True)
    for rel in (STEPS, "data/store/fuel_types.parquet"):
        p = ship_file(s, REPO / rel, s.home_rel(rel))
        print(f"  {rel} -> {p}", flush=True)

    gpu = pick_gpu(s)
    train_args = (
        f"--steps {STEPS} --fuel-types data/store/fuel_types.parquet "
        f"--cache data/policy/_feature_cache.npz "
        f"--arms {args.arms} --seeds {args.seeds} --epochs {args.epochs} "
        f"--device auto --num-workers 8 {args.extra}"
    ).strip()
    workdir = s.workdir.replace("~/", "").rstrip("/")
    script = RUN_SCRIPT.format(workdir=workdir, ts=args.ts, gpu=gpu,
                               python=s.venv_python.replace("~", "$HOME"),
                               args=train_args)
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
