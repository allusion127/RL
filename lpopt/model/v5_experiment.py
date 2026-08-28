"""Pre-registered v5 A/B experiment: four arms, one honest holdout, one table.

Run once at 36-cell curriculum completion.  The comparison is PRE-REGISTERED —
the arms, the seeds, the split, the holdout and the decision metrics are all
fixed here BEFORE any arm trains, so the choice of champion cannot be
rationalized after the fact.

Arms
----
``v4_baseline``      today's schema + today's losses.  The control.
``v5_full``          cond_schema v5 (poison-agnostic channel swap) + physics-prior
                     residual cyclen + quantile heads + promoted max_assembly_burnup.
``v5_minus_shape``   cond_schema v5_noshape — v5 with the k-conv curve-shape
                     channels REMOVED.  This is the ablation that separates "the
                     shape channels carry the poison signal" from "merely dropping
                     the Gd channels helped"; without it a v5 win is unattributable.
``v5_distill``       v5_full plus soft-target distillation from the per-cell best
                     historical teacher over the FULL corpus (no row selection,
                     5-member ensemble preserved, pin-BU targets excluded) — the
                     variant approved on 2026-07-19.

Every arm trains with the SAME base seed, the SAME ensemble size and the SAME
split manifest, so an arm difference is a method difference and not a seed draw.

Decision metrics (all on ``data/splits/S1.json`` ->
``groups.curriculum_val_by_cell``, the honest per-cell holdout that is
permanently excluded from every arm's training fold)
----------------------------------------------------
* **within-cell Spearman** per target (cyclen, f_r) — the ranking skill the
  screening actually consumes, scored per cell and averaged over cells.
* **calibrated MAE** per target — through the full serve path (each arm loads its
  OWN per-cell calibration, so the comparison is calibrated-vs-calibrated, which
  is what will be deployed).
* **legacy-tail check** — cyclen MAE on the high-cyclen legacy Dataset-A bands
  versus the incumbent champion, the guard that caught the v4 tail collapse.
* **P@8 elite precision** — of the 8 patterns an arm ranks best in a cell, how
  many are in the cell's true best 8.  This is the metric that actually decides
  campaign value: a model can lose mid-rank Spearman and still be the better
  screener if its elite set is right.

Nothing here spawns MASTER and nothing here trains locally: ``--dry-run``
validates the plan and prints it, and the live path drives the remote GPU box.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_SPLIT = "S1"
DEFAULT_ENSEMBLE = 5
DEFAULT_BASE_SEED = 20260716
#: the honest holdout group inside the split manifest.
HOLDOUT_GROUP = "curriculum_val_by_cell"
#: targets the decision table scores: ``(name, surrogate column, truth column)``.
DECISION_TARGETS: tuple[tuple[str, int, str], ...] = (
    ("cyclen", 3, "cyclen"),
    ("f_r", 0, "f_r"),
)
#: elite-set size for P@K.
ELITE_K = 8
#: For each target, whether the "elite" end is the HIGH or the LOW tail.
#: cyclen: longer cycle is better.  f_r: lower peaking is better.
ELITE_HIGH_IS_BETTER: dict[str, bool] = {"cyclen": True, "f_r": False}
#: minimum holdout rows in a cell before it is scored at all.
MIN_CELL_ROWS = 8


# --------------------------------------------------------------------------- #
# arm specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArmSpec:
    """One pre-registered arm: a schema + a set of training flags."""

    name: str
    cond_schema: str
    description: str
    physics_prior: bool = False
    quantile_heads: bool = False
    promote_max_asm_bu: bool = False
    distill: bool = False

    def train_args(self, cfg: "ExperimentConfig") -> list[str]:
        """The exact ``python -m lpopt.model.train`` argv for this arm."""
        args = [
            "--ensemble", str(cfg.ensemble),
            "--parallel-members", str(cfg.ensemble),
            "--split", cfg.split,
            "--cond-schema", self.cond_schema,
            "--base-seed", str(cfg.base_seed),
            "--device", "auto",
            "--num-workers", str(cfg.num_workers),
        ]
        if cfg.epochs is not None:
            args += ["--epochs", str(cfg.epochs)]
        args.append("--censor-a-pin-labels" if cfg.censor_a_pin_labels
                    else "--no-censor-a-pin-labels")
        if self.physics_prior:
            args.append("--cyclen-physics-prior")
        if self.quantile_heads:
            args.append("--quantile-heads")
        if self.promote_max_asm_bu:
            args.append("--promote-max-asm-bu")
        if self.distill:
            args += ["--distill-targets", cfg.distill_cache,
                     "--distill-weight", str(cfg.distill_weight)]
        return args


#: The four pre-registered arms, in report order.
ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("v4_baseline", "v4",
            "control: today's schema + today's losses"),
    ArmSpec("v5_full", "v5",
            "poison-agnostic channels + physics prior + quantiles + asm-BU target",
            physics_prior=True, quantile_heads=True, promote_max_asm_bu=True),
    ArmSpec("v5_minus_shape", "v5_noshape",
            "ABLATION: v5 without the k-conv curve-shape channels",
            physics_prior=True, quantile_heads=True, promote_max_asm_bu=True),
    ArmSpec("v5_distill", "v5",
            "v5_full + per-cell teacher soft-target distillation (full corpus)",
            physics_prior=True, quantile_heads=True, promote_max_asm_bu=True,
            distill=True),
)

ARMS_BY_NAME: dict[str, ArmSpec] = {a.name: a for a in ARMS}


@dataclass
class ExperimentConfig:
    """Everything the A/B needs, fixed before the first arm trains."""

    split: str = DEFAULT_SPLIT
    ensemble: int = DEFAULT_ENSEMBLE
    base_seed: int = DEFAULT_BASE_SEED
    epochs: int | None = None
    num_workers: int = 8
    censor_a_pin_labels: bool = True
    store_dir: str = "data/store"
    splits_dir: str = "data/splits"
    models_dir: str = "data/models"
    reports_dir: str = "data/reports"
    deck: str = "lpopt.inp"
    #: incumbent champion, used as the legacy-tail baseline and the default
    #: teacher when a cell has no better historical model.
    champion_dir: str | None = None
    #: ``{cell_key: model_dir}`` map for the distillation arm.
    teacher_map: str | None = None
    distill_cache: str = "data/models/_v5_distill_soft.npz"
    distill_weight: float = 0.3
    arms: tuple[str, ...] = tuple(a.name for a in ARMS)
    library_id: str = "ga80"
    device: str = "cpu"

    def resolved_arms(self) -> list[ArmSpec]:
        return [ARMS_BY_NAME[n] for n in self.arms]


# --------------------------------------------------------------------------- #
# plan construction + validation
# --------------------------------------------------------------------------- #
def build_plan(cfg: ExperimentConfig) -> dict[str, Any]:
    """The full, printable execution plan (no side effects)."""
    return {
        "experiment": "v5_integrated_ab",
        "pre_registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": cfg.split,
        "holdout": f"{cfg.splits_dir}/{cfg.split}.json :: groups.{HOLDOUT_GROUP}",
        "ensemble": cfg.ensemble,
        "base_seed": cfg.base_seed,
        "seeds": [cfg.base_seed + i for i in range(cfg.ensemble)],
        "decision_targets": [t[0] for t in DECISION_TARGETS],
        "decision_metrics": ["within_cell_spearman", "calibrated_mae",
                             "legacy_tail_delta_mae", f"p_at_{ELITE_K}"],
        "champion_dir": cfg.champion_dir,
        "arms": [
            {
                "name": arm.name,
                "cond_schema": arm.cond_schema,
                "description": arm.description,
                "train_argv": ["python", "-m", "lpopt.model.train",
                               *arm.train_args(cfg)],
            }
            for arm in cfg.resolved_arms()
        ],
    }


def validate_plan(cfg: ExperimentConfig) -> list[str]:
    """Every problem that would abort the run, collected (empty list == ok).

    Deliberately exhaustive rather than fail-fast: the point of ``--dry-run`` is
    that the coordinator sees ALL the problems in one pass, at 23:00, instead of
    discovering the second one after the first four-hour arm.
    """
    problems: list[str] = []
    from .featurize import CHANNELS_BY_SCHEMA

    # -- arms ------------------------------------------------------------- #
    for name in cfg.arms:
        if name not in ARMS_BY_NAME:
            problems.append(f"unknown arm {name!r}; have {sorted(ARMS_BY_NAME)}")
    arms = [ARMS_BY_NAME[n] for n in cfg.arms if n in ARMS_BY_NAME]
    for arm in arms:
        if arm.cond_schema not in CHANNELS_BY_SCHEMA:
            problems.append(
                f"arm {arm.name}: unknown cond_schema {arm.cond_schema!r}; "
                f"have {sorted(CHANNELS_BY_SCHEMA)}")
    if cfg.ensemble < 1:
        problems.append(f"ensemble must be >= 1 (got {cfg.ensemble})")

    # -- store + split ----------------------------------------------------- #
    store = Path(cfg.store_dir)
    for fname in ("records.parquet", "fuel_types.parquet"):
        if not (store / fname).is_file():
            problems.append(f"missing store file: {store / fname}")
    split_path = Path(cfg.splits_dir) / f"{cfg.split}.json"
    if not split_path.is_file():
        problems.append(f"missing split manifest: {split_path}")
    else:
        try:
            man = json.loads(split_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"split manifest is not valid JSON: {exc}")
            man = {}
        groups = man.get("groups") or {}
        holdout = groups.get(HOLDOUT_GROUP)
        if not holdout:
            problems.append(
                f"split {cfg.split} carries no groups.{HOLDOUT_GROUP}; the honest "
                f"per-cell holdout is what every arm is scored on")
        else:
            n_cells = len(holdout)
            n_rows = sum(len(v) for v in holdout.values())
            if n_rows < MIN_CELL_ROWS:
                problems.append(
                    f"groups.{HOLDOUT_GROUP} has only {n_rows} rows across "
                    f"{n_cells} cells — too small to decide on")
            # The holdout must be disjoint from the training fold, or the whole
            # comparison is in-sample.  This is the single most important check
            # here, so it is asserted rather than assumed.
            train_ids = set(man.get("train_ids") or ())
            leaked = sorted(
                {r for ids in holdout.values() for r in ids} & train_ids)
            if leaked:
                problems.append(
                    f"LEAKAGE: {len(leaked)} holdout record_ids are also in "
                    f"train_ids (e.g. {leaked[:3]}) — the arms would be scored "
                    f"on rows they trained on")

    # -- shape-channel prerequisite ---------------------------------------- #
    # A v5 arm is meaningless if the k-conv shape columns were never harvested.
    if any(a.cond_schema == "v5" for a in arms):
        ft = store / "fuel_types.parquet"
        if ft.is_file():
            try:
                import pandas as pd
                cols = set(pd.read_parquet(ft).columns)
                missing = {"reactivity_swing_pcm", "depletion_slope_pcm_per_gwd",
                           "bu_peak_gwd", "kinf_peak", "kconv_is_monotone"} - cols
                if missing:
                    problems.append(
                        f"fuel_types is missing k-conv shape columns {sorted(missing)}; "
                        f"the v5 arms have nothing to key on")
            except Exception as exc:      # noqa: BLE001
                problems.append(f"could not read {ft}: {exc}")

    # -- champion (legacy-tail baseline) ----------------------------------- #
    if cfg.champion_dir:
        cd = Path(cfg.champion_dir)
        if not cd.is_dir():
            problems.append(f"champion_dir does not exist: {cd}")
        elif not sorted(cd.glob("member_*")):
            problems.append(f"champion_dir has no member_* checkpoints: {cd}")
    else:
        problems.append(
            "champion_dir is unset — the legacy-tail check needs an incumbent "
            "to compare against")

    # -- distillation arm --------------------------------------------------- #
    if any(a.distill for a in arms):
        if not cfg.teacher_map:
            problems.append(
                "the v5_distill arm needs --teacher-map "
                f"({{cell_key: model_dir}} JSON of the per-cell best teachers, "
                f"or {TEACHER_MAP_AUTO!r} to derive it from {cfg.models_dir})")
        elif cfg.teacher_map == TEACHER_MAP_AUTO:
            # 'auto' defers the (expensive) per-cell teacher selection to launch
            # time; all dry-run can check is that there is something to select
            # FROM, which is still worth failing fast on.
            cands = discover_champions(cfg.models_dir)
            if not cands:
                problems.append(
                    f"--teacher-map auto found no champions with member_* "
                    f"checkpoints under {cfg.models_dir}")
        elif not Path(cfg.teacher_map).is_file():
            problems.append(f"teacher map not found: {cfg.teacher_map}")
        else:
            from .distill import load_teacher_map, validate_teacher_map
            try:
                problems.extend(validate_teacher_map(
                    load_teacher_map(cfg.teacher_map)))
            except ValueError as exc:
                problems.append(f"teacher map is not valid JSON: {exc}")
    return problems


def format_plan(plan: dict[str, Any], problems: Sequence[str]) -> str:
    """Human-readable dry-run report."""
    out: list[str] = []
    out.append("=" * 74)
    out.append("v5 INTEGRATED A/B — PRE-REGISTERED PLAN")
    out.append("=" * 74)
    out.append(f"split            : {plan['split']}")
    out.append(f"honest holdout   : {plan['holdout']}")
    out.append(f"ensemble / seeds : {plan['ensemble']} members, seeds {plan['seeds']}")
    out.append(f"champion (tail)  : {plan['champion_dir']}")
    out.append(f"decision targets : {', '.join(plan['decision_targets'])}")
    out.append(f"decision metrics : {', '.join(plan['decision_metrics'])}")
    out.append("")
    for i, arm in enumerate(plan["arms"], 1):
        out.append(f"[{i}/{len(plan['arms'])}] {arm['name']}  (cond_schema={arm['cond_schema']})")
        out.append(f"      {arm['description']}")
        out.append("      " + " ".join(arm["train_argv"]))
        out.append("")
    if problems:
        out.append("-" * 74)
        out.append(f"VALIDATION FAILED — {len(problems)} problem(s):")
        for p in problems:
            out.append(f"  * {p}")
        out.append("-" * 74)
    else:
        out.append("validation: OK — every arm is runnable as printed.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).correlation
        return float(r) if r is not None else float("nan")
    except Exception:      # pragma: no cover
        import pandas as pd
        return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def precision_at_k(pred: np.ndarray, truth: np.ndarray, k: int = ELITE_K,
                   high_is_better: bool = True) -> float:
    """``|top-k(pred) ∩ top-k(truth)| / k`` — elite-set precision.

    The campaign only ever verifies the handful of candidates the model ranks
    best, so this is the metric that maps directly to MASTER calls saved: a model
    with a mediocre overall Spearman but the right elite set is the better
    screener.  Returns NaN when the cell has fewer than ``k`` scorable rows.
    """
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    if int(ok.sum()) < k:
        return float("nan")
    p, t = pred[ok], truth[ok]
    sign = -1.0 if high_is_better else 1.0
    top_p = set(np.argsort(sign * p, kind="stable")[:k].tolist())
    top_t = set(np.argsort(sign * t, kind="stable")[:k].tolist())
    return len(top_p & top_t) / float(k)


def score_arm(model_dir: str | Path, *, records: Any, holdout: dict[str, list[str]],
              store_dir: str | Path, library_id: str = "ga80",
              device: str = "cpu", champion: Any = None,
              tail_bands: Sequence[Sequence[float]] | None = None,
              tail_feed: int = 121, tail_sample: int = 150,
              seed: int = 0) -> dict[str, Any]:
    """Score ONE arm on the honest per-cell holdout; returns its decision row.

    Every arm is scored through its OWN serve path (``predict``), so each loads
    the per-cell calibration it will actually deploy with — a calibrated-vs-
    calibrated comparison, which is the deployed behaviour being decided on.
    """
    from ..data.schema import unpack_pattern
    from ..vendor.masterrl.domain import CaseKey
    from .model_api import PosValCnnBackend

    model = PosValCnnBackend.from_dir(model_dir, store_dir=store_dir,
                                      library_id=library_id, device=device)
    indexed = records.drop_duplicates("record_id").set_index("record_id")
    per_cell: list[dict[str, Any]] = []
    for cell, ids in sorted(holdout.items()):
        rid = [r for r in ids if r in indexed.index]
        if len(rid) < MIN_CELL_ROWS:
            continue
        sub = indexed.loc[rid].reset_index()
        sub = sub[sub["converged"] == True]          # noqa: E712
        if len(sub) < MIN_CELL_ROWS:
            continue
        pats = [unpack_pattern(str(p)) for p in sub["pattern"]]
        cases = [CaseKey(str(pr), int(fd))
                 for pr, fd in zip(sub["case_pair"], sub["feed"])]
        pred = model.predict(pats, cases)
        import pandas as pd
        for name, col, truth_col in DECISION_TARGETS:
            truth = pd.to_numeric(sub[truth_col], errors="coerce").to_numpy(float)
            pv = np.asarray(pred.mean[:, col], dtype=float)
            ok = np.isfinite(truth) & np.isfinite(pv)
            if int(ok.sum()) < 3:
                continue
            per_cell.append({
                "cell": cell,
                "target": name,
                "n": int(ok.sum()),
                "spearman": _spearman(pv[ok], truth[ok]),
                "mae": float(np.mean(np.abs(pv[ok] - truth[ok]))),
                f"p_at_{ELITE_K}": precision_at_k(
                    pv, truth, ELITE_K, ELITE_HIGH_IS_BETTER.get(name, True)),
            })

    summary: dict[str, Any] = {"model_dir": str(model_dir), "per_cell": per_cell}
    for name, _, _ in DECISION_TARGETS:
        rows = [r for r in per_cell if r["target"] == name]
        summary[name] = {
            "n_cells": len(rows),
            "within_cell_spearman": _nanmean([r["spearman"] for r in rows]),
            "calibrated_mae": _nanmean([r["mae"] for r in rows]),
            f"p_at_{ELITE_K}": _nanmean([r[f"p_at_{ELITE_K}"] for r in rows]),
        }

    # -- legacy high-cyclen tail no-regression vs the incumbent champion ----
    summary["legacy_tail"] = {"note": "no champion supplied"}
    if champion is not None and tail_bands:
        from ..curriculum import score_legacy_tail_no_regression
        summary["legacy_tail"] = score_legacy_tail_no_regression(
            champion, model, records, bands=tail_bands, feed=tail_feed,
            sample_per_band=tail_sample, seed=seed)
    return summary


def _nanmean(vals: Sequence[float]) -> float:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def decision_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the per-arm decision table as markdown."""
    lines = ["# v5 integrated A/B — decision table", ""]
    hdr = (f"| arm | target | within-cell Spearman | calibrated MAE | "
           f"P@{ELITE_K} | n cells |")
    lines += [hdr, "|---|---|--:|--:|--:|--:|"]
    for arm, res in results.items():
        for name, _, _ in DECISION_TARGETS:
            s = res.get(name, {})
            lines.append(
                f"| {arm} | {name} | {_fmt(s.get('within_cell_spearman'))} | "
                f"{_fmt(s.get('calibrated_mae'))} | "
                f"{_fmt(s.get(f'p_at_{ELITE_K}'))} | {s.get('n_cells', 0)} |")
    lines += ["", "## legacy high-cyclen tail (vs incumbent champion)", "",
              "| arm | pass | worst band MAE increase [EFPD] |", "|---|---|--:|"]
    for arm, res in results.items():
        t = res.get("legacy_tail") or {}
        # score_legacy_tail_no_regression reports ``worst_mae_increase``.
        worst = t.get("worst_mae_increase", t.get("worst_increase"))
        lines.append(f"| {arm} | {t.get('pass', 'n/a')} | {_fmt(worst)} |")
    return "\n".join(lines)


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(f) else f"{f:.{nd}f}"


# --------------------------------------------------------------------------- #
# arm-dir manifest (so --score-only auto never needs a hand-written file)
# --------------------------------------------------------------------------- #
#: sentinel for ``--score-only``: resolve arm dirs from the runner's manifest.
SCORE_ONLY_AUTO = "auto"


def _arm_manifest_path(cfg: ExperimentConfig) -> Path:
    return Path(cfg.reports_dir) / "v5_arm_dirs.json"


def _write_arm_manifest(cfg: ExperimentConfig, arm_dirs: dict[str, str]) -> Path:
    p = _arm_manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp")
    tmp.write_text(json.dumps(arm_dirs, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def _load_arm_manifest(cfg: ExperimentConfig) -> dict[str, str]:
    p = _arm_manifest_path(cfg)
    if not p.is_file():
        return {}
    return {str(k): str(v) for k, v in
            json.loads(p.read_text(encoding="utf-8")).items()}


# --------------------------------------------------------------------------- #
# execution (remote GPU)
# --------------------------------------------------------------------------- #
def run_experiment(cfg: ExperimentConfig, *, dry_run: bool = True,
                   log=print) -> int:
    """Validate, print, and (unless ``dry_run``) drive the four arms remotely."""
    plan = build_plan(cfg)
    problems = validate_plan(cfg)
    log(format_plan(plan, problems))
    if problems:
        return 1
    if dry_run:
        log("\n--dry-run: nothing launched.")
        return 0

    from .. import remote as remote_mod
    from ..config import load_config

    full = load_config(Path(cfg.deck))
    import dataclasses
    s = remote_mod.RemoteSettings(**dataclasses.asdict(full.remote))

    # Build the per-cell teacher map + distillation cache once, locally, before
    # any arm launches (so a failure here costs no GPU time).
    if any(a.distill for a in cfg.resolved_arms()):
        if cfg.teacher_map == TEACHER_MAP_AUTO:
            auto_path = Path(cfg.models_dir) / "_v5_teacher_map.json"
            log("=== deriving the per-cell best teacher map ===")
            # Exclude any arm output already pulled (a resume/re-run): an arm dir
            # must never become a teacher of the distillation student.
            exclude = [Path(d).name for d in _load_arm_manifest(cfg).values()]
            build_teacher_map(cfg, out_path=auto_path, exclude=exclude, log=log)
            cfg.teacher_map = str(auto_path)
        _build_distill_cache(cfg, log=log)

    log("=== pushing source + data to the remote box ===")
    remote_mod.push(s, install=True)

    # Ship the built distillation cache to the remote workdir BEFORE launching the
    # distill arm.  push() ships data/store + data/splits but not data/models, so
    # the cache the runner just built locally would be absent and the remote arm
    # would die with FileNotFoundError on the relative --distill-targets path
    # (the exact first-run failure).  Shipped by cat-over-ssh (scp mangles this
    # repo's non-ASCII local path).
    if any(a.distill for a in cfg.resolved_arms()) and Path(cfg.distill_cache).is_file():
        remote_rel = s.home_rel(*Path(cfg.distill_cache).as_posix().split("/"))
        landed = remote_mod.ship_file(s, Path(cfg.distill_cache), remote_rel)
        log(f"=== shipped distillation cache -> {landed} ===")

    results: dict[str, dict[str, Any]] = {}
    pulled: dict[str, str] = _load_arm_manifest(cfg)   # resume-friendly
    for arm in cfg.resolved_arms():
        log(f"=== launching arm {arm.name} ===")
        tr = remote_mod.train(s, arm.train_args(cfg))
        ts = tr["ts"]
        while True:
            st = remote_mod.status(s, ts)
            if st.get("state") == "done":
                break
            if st.get("state") == "failed":
                raise RuntimeError(f"arm {arm.name} failed on the remote box "
                                   f"(run {ts}); see its train.log")
            time.sleep(60)
        pl = remote_mod.pull(s, ts)
        pulled[arm.name] = pl["dest"]
        # Persist the arm->dir map incrementally, so a crash mid-run still leaves
        # a manifest '--score-only auto' can read (the coordinator had to
        # hand-write this file).
        _write_arm_manifest(cfg, pulled)
        log(f"=== arm {arm.name} pulled to {pl['dest']} ===")

    _write_arm_manifest(cfg, pulled)
    results = score_all(cfg, pulled, log=log)
    _emit_report(cfg, plan, pulled, results, log=log)
    return 0


def _emit_report(cfg: ExperimentConfig, plan: dict[str, Any],
                 arm_dirs: dict[str, str], results: dict[str, dict[str, Any]],
                 log=print) -> Path:
    out = Path(cfg.reports_dir) / "v5_experiment.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(decision_table(results), encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps({"plan": plan, "arms": arm_dirs, "results": results},
                   indent=2, sort_keys=True, default=str), encoding="utf-8")
    log(f"=== decision table written to {out} ===")
    return out


def score_all(cfg: ExperimentConfig, arm_dirs: dict[str, str],
              log=print) -> dict[str, dict[str, Any]]:
    """Score every trained arm on the one honest holdout."""
    from ..config import load_config
    from ..data.store import StoreReader
    from .model_api import PosValCnnBackend
    from .splits import SplitManifest

    records = StoreReader(cfg.store_dir).records
    man = SplitManifest.from_json(Path(cfg.splits_dir) / f"{cfg.split}.json")
    holdout = man.groups.get(HOLDOUT_GROUP, {})
    champion = None
    tail_bands = None
    if cfg.champion_dir:
        champion = PosValCnnBackend.from_dir(
            cfg.champion_dir, store_dir=cfg.store_dir,
            library_id=cfg.library_id, device=cfg.device)
        try:
            tail_bands = load_config(Path(cfg.deck)).curriculum.gate_tail_bands
        except Exception:      # noqa: BLE001 - fall back to the default bands
            tail_bands = [[700.0, 720.0], [680.0, 700.0]]
    results: dict[str, dict[str, Any]] = {}
    for name, d in arm_dirs.items():
        log(f"=== scoring arm {name} ===")
        results[name] = score_arm(
            d, records=records, holdout=holdout, store_dir=cfg.store_dir,
            library_id=cfg.library_id, device=cfg.device, champion=champion,
            tail_bands=tail_bands)
    return results


#: sentinel for ``--teacher-map``: derive the per-cell best teacher automatically.
TEACHER_MAP_AUTO = "auto"


def discover_champions(models_dir: str | Path) -> list[Path]:
    """Every historical model dir under ``models_dir`` holding member checkpoints."""
    root = Path(models_dir)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and sorted(d.glob("member_*")))


def build_teacher_map(cfg: ExperimentConfig, *, out_path: str | Path,
                      candidates: Sequence[str | Path] | None = None,
                      target: str = "cyclen", exclude: Sequence[str] | None = None,
                      log=print) -> dict[str, str]:
    """Pick, per holdout cell, the historical champion that ranks it best.

    "Best" is the highest within-cell Spearman on ``target`` over that cell's
    honest holdout rows — the same metric the decision table reports, so the
    teacher selection and the final judgement agree on what "good on this cell"
    means.  Each candidate is scored through its OWN serve path.

    ``exclude`` drops candidate dirs by BASENAME — always pass this experiment's
    own arm output dirs.  A teacher must be a HISTORICAL champion; letting an arm
    (v5_full, v5_minus_shape, ...) become a teacher makes the distill student
    learn from the very models it is adjudicated against.  A hard self-check
    RAISES if any selected teacher is in ``exclude``, so the leak cannot pass
    silently (it did once: the first teacher map assigned 3 cells to arm dirs).

    This is the expensive part of the distillation arm (one forward pass per
    champion per cell), so it is a separate, cacheable step: build the map once,
    then pass the file to every subsequent run.
    """
    from ..data.schema import unpack_pattern
    from ..data.store import StoreReader
    from ..vendor.masterrl.domain import CaseKey
    from .model_api import PosValCnnBackend
    from .splits import SplitManifest
    import pandas as pd

    excl = {str(x) for x in (exclude or ())}
    cands = [Path(c) for c in (candidates or discover_champions(cfg.models_dir))
             if Path(c).name not in excl]
    if not cands:
        raise FileNotFoundError(
            f"no eligible champions with member_* under {cfg.models_dir} "
            f"(after excluding {sorted(excl)})")
    records = StoreReader(cfg.store_dir).records
    man = SplitManifest.from_json(Path(cfg.splits_dir) / f"{cfg.split}.json")
    holdout = man.groups.get(HOLDOUT_GROUP, {})
    indexed = records.drop_duplicates("record_id").set_index("record_id")
    col = dict((n, c) for n, c, _ in DECISION_TARGETS)[target]

    best: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    for cand in cands:
        try:
            model = PosValCnnBackend.from_dir(
                cand, store_dir=cfg.store_dir, library_id=cfg.library_id,
                device=cfg.device)
        except Exception as exc:      # noqa: BLE001 - skip an unloadable champion
            log(f"  skip {cand.name}: {exc}")
            continue
        for cell, ids in sorted(holdout.items()):
            rid = [r for r in ids if r in indexed.index]
            if len(rid) < MIN_CELL_ROWS:
                continue
            sub = indexed.loc[rid].reset_index()
            sub = sub[sub["converged"] == True]          # noqa: E712
            if len(sub) < MIN_CELL_ROWS:
                continue
            pats = [unpack_pattern(str(p)) for p in sub["pattern"]]
            cases = [CaseKey(str(pr), int(fd))
                     for pr, fd in zip(sub["case_pair"], sub["feed"])]
            try:
                pred = model.predict(pats, cases)
            except Exception:      # noqa: BLE001
                continue
            truth = pd.to_numeric(sub[target], errors="coerce").to_numpy(float)
            pv = np.asarray(pred.mean[:, col], dtype=float)
            ok = np.isfinite(truth) & np.isfinite(pv)
            r = _spearman(pv[ok], truth[ok]) if int(ok.sum()) >= 3 else float("nan")
            if not np.isfinite(r):
                continue
            prev = scores.setdefault(cell, {})
            if r > prev.get("_best", -np.inf):
                prev["_best"] = r
                best[cell] = str(cand)
        log(f"  scored {cand.name}")
    # hard self-check: no chosen teacher may be an excluded (arm) dir.
    leaks = {c: d for c, d in best.items() if Path(d).name in excl}
    if leaks:
        raise ValueError(
            f"teacher map leak: {len(leaks)} cell(s) selected an EXCLUDED dir as "
            f"teacher (e.g. {sorted(leaks.items())[:2]}); an experiment arm must "
            f"never teach its own distillation student")
    out = {"schema": "v5_teacher_map_v1", "target": target,
           "split": cfg.split, "n_candidates": len(cands),
           "excluded": sorted(excl),
           "best_spearman": {k: v["_best"] for k, v in scores.items()},
           "teachers": best}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    log(f"=== teacher map written to {p} ({len(best)} cells) ===")
    return best


def _build_distill_cache(cfg: ExperimentConfig, log=print) -> str:
    """Build the per-cell teacher soft-target cache for the distillation arm.

    Rows are keyed by their ``campaign`` — the SAME identity the teacher map is
    keyed by (``curriculum_val_by_cell`` groups rows by ``campaign == cell_id``,
    see ``splits.make_curriculum_split``).  The earlier decoy keyed rows by
    ``cyclen_cell_key`` (``feed=121|ebin=5.4``) while the teacher map was keyed by
    curriculum cell-id (``5-5.25_f101``): disjoint namespaces, zero matches, a
    silent all-zero cache.  ``build_soft_targets`` now hard-errors on that.
    """
    from ..data.store import StoreReader
    from .dataset_torch import targets_for
    from .distill import build_soft_targets, load_teacher_map
    from .splits import SplitManifest

    records = StoreReader(cfg.store_dir).records
    man = SplitManifest.from_json(Path(cfg.splits_dir) / f"{cfg.split}.json")
    train_ids = set(man.record_ids("train"))
    df = records[records["record_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    # campaign is the curriculum-cell identity for a Dataset-P row; legacy A/B
    # rows carry a non-cell campaign, fall in no teacher cell, and correctly get
    # no soft target (they train on hard labels — "full corpus, no row selection").
    keys = df["campaign"].astype(str).tolist()
    teachers = load_teacher_map(cfg.teacher_map)
    log(f"=== building distillation soft targets for {len(df)} train rows "
        f"from {len(teachers)} per-cell teachers ===")
    build_soft_targets(
        df, teachers, cell_keys=keys,
        target_names=targets_for(True), store_dir=cfg.store_dir,
        device=cfg.device, library_id=cfg.library_id,
        out_path=cfg.distill_cache)
    return cfg.distill_cache


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the v5-experiment options on ``ap``.

    Shared by the module's own ``main`` and by the ``lpopt v5-experiment``
    subparser, so ``lpopt v5-experiment --dry-run`` and
    ``python -m lpopt.model.v5_experiment --dry-run`` accept the identical flags
    (an ``argparse.REMAINDER`` passthrough would swallow a LEADING option before
    the subcommand ever saw it).
    """
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the config and print the plan; launch nothing")
    ap.add_argument("--input", "-i", default="lpopt.inp", help="campaign TOML deck")
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--ensemble", type=int, default=DEFAULT_ENSEMBLE)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--store-dir", default="data/store")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--reports-dir", default="data/reports")
    ap.add_argument("--champion-dir", default=None,
                    help="incumbent champion (legacy-tail baseline); "
                         "defaults to the deck's [model] model_dir")
    ap.add_argument("--teacher-map", default=None,
                    help="{cell_key: model_dir} JSON for the distillation arm, "
                         f"or {TEACHER_MAP_AUTO!r} to derive it from --models-dir")
    ap.add_argument("--models-dir", default="data/models",
                    help="historical champion root (teacher-map auto discovery)")
    ap.add_argument("--build-teacher-map", default=None,
                    help="derive + write the per-cell teacher map to this path, "
                         "then exit (no training)")
    ap.add_argument("--distill-weight", type=float, default=0.3)
    ap.add_argument("--arms", default=None,
                    help=f"comma-separated subset of {[a.name for a in ARMS]}")
    ap.add_argument("--score-only", default=None,
                    help="skip training, just score: 'auto' reads the runner's "
                         "v5_arm_dirs.json manifest, or pass a {arm: model_dir} "
                         "JSON path")
    return ap


def build_arg_parser() -> argparse.ArgumentParser:
    return add_arguments(argparse.ArgumentParser(
        prog="lpopt v5-experiment",
        description="pre-registered v5 integrated A/B on the remote GPU"))


def _champion_from_deck(deck: str | Path) -> str | None:
    """``[model] model_dir`` from the campaign deck (the deployed champion).

    Used as the default legacy-tail baseline so the coordinator does not have to
    retype a timestamped path at launch time; ``--champion-dir`` still wins, and
    the dry-run prints whichever was resolved so it can be eyeballed.
    """
    try:
        from ..config import load_config
        d = load_config(Path(deck)).model.model_dir
    except Exception:      # noqa: BLE001 - a missing/invalid deck is not fatal here
        return None
    return d if d and Path(d).is_dir() else None


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        split=args.split, ensemble=args.ensemble, base_seed=args.base_seed,
        epochs=args.epochs, store_dir=args.store_dir, splits_dir=args.splits_dir,
        reports_dir=args.reports_dir, deck=args.input,
        champion_dir=args.champion_dir or _champion_from_deck(args.input),
        teacher_map=args.teacher_map,
        distill_weight=args.distill_weight,
        models_dir=getattr(args, "models_dir", "data/models"),
    )
    if args.arms:
        cfg.arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    return cfg


def resolve_score_only(cfg: ExperimentConfig, spec: str) -> dict[str, str]:
    """Resolve ``--score-only`` to an ``{arm: model_dir}`` map.

    ``auto`` reads the runner's own ``v5_arm_dirs.json`` manifest (written
    incrementally as each arm is pulled); a path reads that JSON file directly.
    Both raise a clear error rather than a bare KeyError/FileNotFoundError so a
    user never has to hand-write the manifest (the coordinator's C(ii) request).
    """
    if spec == SCORE_ONLY_AUTO:
        manifest = _load_arm_manifest(cfg)
        if not manifest:
            raise FileNotFoundError(
                f"--score-only auto found no arm manifest at "
                f"{_arm_manifest_path(cfg)}; run the experiment first, or pass an "
                f"explicit {{arm: model_dir}} JSON path.")
        return manifest
    p = Path(spec)
    if not p.is_file():
        raise FileNotFoundError(
            f"--score-only {spec!r} is neither 'auto' nor an existing JSON file")
    return {str(k): str(v) for k, v in
            json.loads(p.read_text(encoding="utf-8")).items()}


def run_from_args(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    if getattr(args, "build_teacher_map", None):
        build_teacher_map(cfg, out_path=args.build_teacher_map)
        return 0
    if args.score_only:
        arm_dirs = resolve_score_only(cfg, args.score_only)
        results = score_all(cfg, arm_dirs)
        print(decision_table(results))
        _emit_report(cfg, build_plan(cfg), arm_dirs, results)
        return 0
    return run_experiment(cfg, dry_run=bool(args.dry_run))


def main(argv: Sequence[str] | None = None) -> int:
    return run_from_args(build_arg_parser().parse_args(argv))


__all__ = [
    "ARMS", "ARMS_BY_NAME", "ArmSpec", "DECISION_TARGETS", "ELITE_K",
    "HOLDOUT_GROUP", "ExperimentConfig", "add_arguments", "build_arg_parser",
    "build_plan", "build_teacher_map", "config_from_args", "decision_table",
    "format_plan", "main", "precision_at_k", "resolve_score_only",
    "run_experiment", "run_from_args", "score_all", "score_arm",
    "SCORE_ONLY_AUTO", "TEACHER_MAP_AUTO", "validate_plan",
]


if __name__ == "__main__":          # pragma: no cover
    raise SystemExit(main())
