"""Watch the GPU box for finished A/B arms, pull them, and score them.

``python -m lpopt.model.ab_watch --poll 300``

The launch chain fires arms sequentially, so their run timestamps are not known
in advance.  Rather than requiring a hand-maintained ts->arm mapping (which goes
stale the moment a run is relaunched), this identifies each arm from the training
flags recorded in its own ``run.sh``: the arm table is a signature over
(cond_schema, width, n_blocks, head_hidden, map decoder, prior residual, spectral
weight), which is exactly what distinguishes the arms by construction.

A run is processed once, when it first shows ``DONE``.  Already-scored arms are
skipped by consulting the results file, so the watcher is safe to restart.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import ab_eval as M
from .ab_score import render_markdown, score_arm, update_results

#: Flags EVERY hires arm carries, and which historical runs do not.  Without
#: these, B1's shape (v5 / 160 / 6 / linear) also matches the 20260721_105824
#: champion and the frozen-trunk 20260724_213535 -- both would be mis-scored as
#: B1.  ``init_from``/``freeze_trunk_cyclen`` absent is the load-bearing part:
#: every arm trains its trunk, which is the whole point of the B1 control.
COMMON_SIGNATURE: dict[str, Any] = {
    "map_peak_weight": 2.0,
    "init_from": None,
    "freeze_trunk_cyclen": False,
}
#: Runs older than this cannot belong to the campaign (belt and braces on top of
#: COMMON_SIGNATURE, so a future recipe reuse cannot silently re-label history).
CAMPAIGN_START_TS = "20260725_000000"

#: Arm signature table — must mirror ``launch_hires_ab.ps1``.  Keys are the flags
#: that actually differ between arms; COMMON_SIGNATURE is added to each.
ARM_SIGNATURES: dict[str, dict[str, Any]] = {
    "B1": {"cond_schema": "v5", "width": 160, "n_blocks": 6,
           "map_decoder": "linear", "map_prior_residual": False,
           "map_spectral_weight": 0.0},
    "A1": {"cond_schema": "v5", "width": 160, "n_blocks": 6,
           "map_decoder": "multiscale", "map_prior_residual": False,
           "map_spectral_weight": 0.0},
    "A2": {"cond_schema": "v6_prior", "width": 160, "n_blocks": 6,
           "map_decoder": "linear", "map_prior_residual": True,
           "map_spectral_weight": 0.0},
    "A3": {"cond_schema": "v5", "width": 160, "n_blocks": 6,
           "map_decoder": "linear", "map_prior_residual": False,
           "map_spectral_weight": 0.3},
    "A4": {"cond_schema": "v6_contrast", "width": 160, "n_blocks": 6,
           "map_decoder": "linear", "map_prior_residual": False,
           "map_spectral_weight": 0.0},
    "A5": {"cond_schema": "v5", "width": 256, "n_blocks": 10,
           "map_decoder": "linear", "map_prior_residual": False,
           "map_spectral_weight": 0.0},
    "A6": {"cond_schema": "v6", "width": 224, "n_blocks": 8,
           "map_decoder": "multiscale", "map_prior_residual": True,
           "map_spectral_weight": 0.3},
}


def parse_train_args(text: str) -> dict[str, Any]:
    """Extract the arm-distinguishing flags from a ``run.sh`` (or a raw cmdline)."""
    m = re.search(r"-m\s+lpopt\.model\.train\s+(.*?)(?:>\s*\"?\$RUN|$)", text, re.S)
    argv = shlex.split(m.group(1) if m else text)
    out: dict[str, Any] = {
        "cond_schema": "v3", "width": 112, "n_blocks": 6, "head_hidden": 256,
        "map_decoder": "linear", "map_prior_residual": False,
        "map_spectral_weight": 0.0, "map_peak_weight": 0.0,
        "init_from": None, "freeze_trunk_cyclen": False,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--map-prior-residual":
            out["map_prior_residual"] = True
        elif a == "--freeze-trunk-cyclen":
            out["freeze_trunk_cyclen"] = True
        elif a == "--init-from" and i + 1 < len(argv):
            out["init_from"] = argv[i + 1]
            i += 1
        elif a == "--map-peak-weight" and i + 1 < len(argv):
            out["map_peak_weight"] = float(argv[i + 1])
            i += 1
        elif a in ("--cond-schema", "--map-decoder") and i + 1 < len(argv):
            out[a[2:].replace("-", "_")] = argv[i + 1]
            i += 1
        elif a in ("--width", "--n-blocks", "--head-hidden") and i + 1 < len(argv):
            out[a[2:].replace("-", "_")] = int(argv[i + 1])
            i += 1
        elif a == "--map-spectral-weight" and i + 1 < len(argv):
            out["map_spectral_weight"] = float(argv[i + 1])
            i += 1
        i += 1
    return out


def identify_arm(text: str, *, ts: str | None = None) -> str | None:
    """Arm label for a ``run.sh``, or ``None`` if it matches no campaign arm.

    ``ts`` (the run timestamp) is rejected when it predates the campaign, so a
    historical run that happens to share an arm's hyperparameters is never
    mistaken for one.
    """
    if ts is not None and str(ts) < CAMPAIGN_START_TS:
        return None
    got = parse_train_args(text)
    for label, sig in ARM_SIGNATURES.items():
        full = {**COMMON_SIGNATURE, **sig}
        if all(got.get(k) == v for k, v in full.items()):
            return label
    return None


# --------------------------------------------------------------------------- #
# remote discovery
# --------------------------------------------------------------------------- #
@dataclass
class RemoteRun:
    ts: str
    arm: str | None
    done: bool
    rc: str | None


def discover_runs(settings: Any, registry: dict[str, str] | None = None,
                  ) -> list[RemoteRun]:
    """List remote runs with their completion markers and resolved arm label."""
    from ..remote import _clean, run_ssh

    cmd = (f"for d in {settings.workdir}/runs/*/; do "
           f"ts=$(basename $d); "
           f"mk=$( [ -f $d/DONE ] && echo DONE || ([ -f $d/FAILED ] && echo FAILED || echo RUN) ); "
           f"rc=$(cat $d/rc 2>/dev/null || echo -); "
           f"echo \"===$ts|$mk|$rc\"; cat $d/run.sh 2>/dev/null; done")
    cp = run_ssh(settings, cmd, timeout=120)
    text = _clean(cp.stdout.decode(errors="replace"))
    runs: list[RemoteRun] = []
    for block in text.split("===")[1:]:
        header, _, body = block.partition("\n")
        parts = header.strip().split("|")
        if len(parts) < 3:
            continue
        ts, mk, rc = parts[0], parts[1], parts[2]
        arm = (registry or {}).get(ts) or identify_arm(body, ts=ts)
        runs.append(RemoteRun(ts=ts, arm=arm, done=(mk == "DONE"),
                              rc=(None if rc in ("-", "") else rc)))
    return runs


#: Explicit label -> run-timestamp registry.  Signature matching is the fallback;
#: an explicit registration is authoritative because it survives a recipe that
#: two arms happen to share and it pins WHICH relaunch counts.
DEFAULT_REGISTRY = "data/reports/hires_ab_runs.json"


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, str]:
    """``{ts: label}`` (inverted from the on-disk ``{label: ts}``)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(ts): str(label) for label, ts in (doc.get("arms") or {}).items()}


def save_registry(mapping: dict[str, str], path: str | Path = DEFAULT_REGISTRY,
                  ) -> dict[str, str]:
    """Merge ``{label: ts}`` into the registry file and return the full map."""
    p = Path(path)
    doc: dict[str, Any] = {"schema": "hires_ab_runs_v1", "arms": {}}
    if p.exists():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    doc.setdefault("arms", {}).update(mapping)
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc["arms"]


def scored_labels(results: str | Path = M.DEFAULT_RESULTS) -> dict[str, str]:
    """``{label: model_dir}`` already present in the results file."""
    p = Path(results)
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {k: v.get("model_dir", "") for k, v in (doc.get("arms") or {}).items()}


# --------------------------------------------------------------------------- #
# one sweep
# --------------------------------------------------------------------------- #
def sweep(*, input_deck: str = "lpopt.inp", results: str = M.DEFAULT_RESULTS,
          markdown: str = "data/reports/hires_ab_results.md",
          registry: str = DEFAULT_REGISTRY,
          folds: Sequence[str] = ("C", "B"), device: str = "cpu",
          champion: str | None = M.DEFAULT_CHAMPION, rescore: bool = False,
          dry_run: bool = False, verbose: bool = True) -> list[str]:
    """Pull + score every finished, unscored arm.  Returns the labels handled.

    Arms run in PARALLEL on the box, so a single sweep routinely has to handle
    several completions at once.  Each arm is isolated in its own try/except: a
    failed pull or a corrupt checkpoint must not deny the other six their score,
    and the next sweep retries whatever failed.
    """
    from ..remote import RemoteSettings, pull

    s = RemoteSettings.from_input(input_deck)
    reg = load_registry(registry)
    runs = discover_runs(s, reg)
    already = scored_labels(results)
    handled: list[str] = []
    if verbose:
        for r in runs:
            if r.arm is None and str(r.ts) < CAMPAIGN_START_TS:
                continue                      # don't spam the log with history
            state = "DONE" if r.done else "running"
            print(f"  {r.ts}  {state:8s} arm={r.arm or '?':3s} rc={r.rc or '-'}",
                  flush=True)
    pending = [r for r in runs if r.done and r.arm is not None
               and (rescore or r.arm not in already)]
    if verbose and pending:
        print(f"  pending: {', '.join(r.arm for r in pending)}", flush=True)
    for r in pending:
        dest = Path("data/models") / r.ts
        if dry_run:
            print(f"[dry] would pull {r.ts} -> {dest} and score as {r.arm}",
                  flush=True)
            handled.append(r.arm)
            continue
        try:
            if not (dest / "ensemble.json").exists():
                pull(s, r.ts)
            entry = score_arm(r.arm, dest, folds=tuple(folds), device=device,
                              champion_dir=champion, verbose=verbose)
            doc = update_results(entry, results)
            Path(markdown).parent.mkdir(parents=True, exist_ok=True)
            Path(markdown).write_text(render_markdown(doc), encoding="utf-8")
            handled.append(r.arm)
            if verbose:
                print(f"[{r.arm}] scored -> {results} / {markdown}", flush=True)
        except Exception as exc:              # one bad arm must not block the rest
            print(f"[{r.arm}] SCORING FAILED ({type(exc).__name__}: {exc}); "
                  f"will retry next sweep", flush=True)
    return handled


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lpopt.model.ab_watch")
    ap.add_argument("--input", default="lpopt.inp")
    ap.add_argument("--results", default=M.DEFAULT_RESULTS)
    ap.add_argument("--markdown", default="data/reports/hires_ab_results.md")
    ap.add_argument("--folds", default="C,B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--champion", default=M.DEFAULT_CHAMPION)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--poll", type=int, default=0,
                    help="seconds between sweeps (0 = one sweep and exit)")
    ap.add_argument("--max-sweeps", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--register", action="append", default=[], metavar="LABEL=TS",
                    help="pin an arm to a run timestamp (repeatable); overrides "
                         "run.sh signature matching")
    args = ap.parse_args(argv)

    if args.register:
        mapping = {}
        for spec in args.register:
            if "=" not in spec:
                ap.error(f"--register must be LABEL=TS (got {spec!r})")
            label, ts = spec.split("=", 1)
            mapping[label] = ts
        full = save_registry(mapping, args.registry)
        print(f"registry -> {args.registry}: "
              + ", ".join(f"{k}={v}" for k, v in sorted(full.items())), flush=True)

    import torch
    torch.set_num_threads(max(1, int(args.threads)))

    n = 0
    while True:
        n += 1
        print(f"=== sweep {n} {time.strftime('%H:%M:%S')} ===", flush=True)
        try:
            done = sweep(input_deck=args.input, results=args.results,
                         markdown=args.markdown, registry=args.registry,
                         folds=tuple(f for f in args.folds.split(",") if f),
                         device=args.device,
                         champion=(args.champion or None),
                         rescore=args.rescore, dry_run=args.dry_run)
            if done:
                print(f"scored this sweep: {', '.join(done)}", flush=True)
        except Exception as exc:                       # keep the watcher alive
            print(f"[sweep error] {type(exc).__name__}: {exc}", flush=True)
        if not args.poll:
            return 0
        if args.max_sweeps and n >= args.max_sweeps:
            return 0
        time.sleep(int(args.poll))


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
