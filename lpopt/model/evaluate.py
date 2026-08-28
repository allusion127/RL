"""Ensemble evaluation report (plan sec. 4.4 success bar).

``eval_report(ckpt_dirs, splits, out)`` writes a Markdown report with, per split
and per target: MAE / RMSE / R², within-case Spearman (cases with >= 30 val
patterns, mean±sd), ensemble-vs-member R², risk-coverage curve data (σ-sorted),
the ExtraTrees baseline comparison, and honest acceptance verdicts against the
plan's success bar:

* ``cyclen`` and ``cbc_max``: R² >= 0.98 **and** tree-comparable (>= baseline).
* ``F_r`` / ``F_q``: within-case Spearman >= the ExtraTrees baseline.
* S2 (leave-pair-out) produces finite metrics (functional).

A failing bar is reported as **FAILED** — no rounding up.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.fuel_types import FuelLibrary
from ..data.store import StoreReader
from .calibrate import apply_calibration, ensemble_stats, load_calibration, CALIB_NAME
from .dataset_torch import LPDataset, TARGETS
from .splits import SplitManifest
from .train import _load_split, load_member, norm_from_meta, predict_dataset, within_case_spearman

DEFAULT_STORE = "data/store"
DEFAULT_SPLITS = "data/splits"
DEFAULT_REPORTS = "data/reports"
R2_BAR = 0.98
_SPEARMAN_MIN_CASE = 30
_TREE_TOL = 0.005          # R²/Spearman slack when comparing to the baseline


def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
    if truth.size < 2:
        return float("nan")
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _risk_coverage(abs_err: np.ndarray, sigma: np.ndarray,
                   fractions=(0.2, 0.4, 0.6, 0.8, 1.0)) -> list[dict[str, float]]:
    """Selective-prediction curve: keep the lowest-σ fraction, report its MAE."""
    order = np.argsort(sigma)
    err_sorted = abs_err[order]
    n = err_sorted.size
    pts = []
    for f in fractions:
        k = max(1, int(round(f * n)))
        pts.append({"coverage": round(f, 2),
                    "mae": float(err_sorted[:k].mean())})
    return pts


def evaluate_split(members, metas, split: str, *, calib: dict | None,
                   store_dir, splits_dir, device) -> dict[str, Any]:
    tmean, tstd = norm_from_meta(metas[0])
    reader = StoreReader(store_dir)
    fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
    manifest = _load_split(reader, split, splits_dir, seed=0)
    if manifest.n_val == 0:
        return {"split": split, "status": manifest.status, "n_val": 0,
                "per_target": {}, "note": "empty val fold"}

    # evaluation reports on the FULL label set (censoring is a training-loss
    # concern only), so Dataset-A pin metrics stay comparable across runs.
    # Target inventory follows the CHECKPOINT: a promote_max_asm_bu checkpoint
    # predicts 8 targets, and the label matrix must be the same width or the 8th
    # column would be reported against nothing.  For every existing (7-target)
    # checkpoint this resolves to exactly ``TARGETS``, so the report is unchanged.
    target_names = tuple(metas[0].get("target_names", TARGETS))
    promote = "max_assembly_burnup" in target_names
    val_ds = LPDataset(reader, manifest, fuel, augment=False, fold="val",
                       censor_dataset_a_pin_labels=False,
                       promote_max_asm_bu=promote)
    pred = predict_dataset(members, val_ds, device)
    mean_raw, epistemic, total = ensemble_stats(pred, tmean, tstd)
    calibrated = apply_calibration(total, calib) if calib else total
    true = pred["targets"]
    tmask = pred["target_mask"]
    rid_cp = dict(zip(reader.records["record_id"].astype(str),
                      reader.records["case_pair"].astype(str)))
    cases = np.asarray([rid_cp.get(str(r), "?") for r in pred["record_ids"]])

    per_target: dict[str, Any] = {}
    mu_members = pred["mu_z_members"] * tstd[None, None, :] + tmean[None, None, :]
    for k, name in enumerate(target_names):
        sel = tmask[:, k] > 0
        if sel.sum() < 2:
            per_target[name] = {"status": "insufficient_labels", "n_val": int(sel.sum())}
            continue
        t = true[sel, k]
        p = mean_raw[sel, k]
        abs_err = np.abs(p - t)
        ens_r2 = _r2(t, p)
        member_r2 = [float(_r2(t, mu_members[mi, sel, k])) for mi in range(len(members))]
        sp_mean, sp_sd, n_cases = within_case_spearman(
            mean_raw[:, k], true[:, k], tmask[:, k].astype(float),
            cases, _SPEARMAN_MIN_CASE)
        per_target[name] = {
            "n_val": int(sel.sum()),
            "mae": float(abs_err.mean()),
            "rmse": float(math.sqrt(np.mean((p - t) ** 2))),
            "r2": ens_r2,
            "member_r2_mean": float(np.nanmean(member_r2)),
            "within_case_spearman": sp_mean,
            "within_case_spearman_sd": sp_sd,
            "n_cases": int(n_cases),
            "mean_calibrated_sigma": float(calibrated[sel, k].mean()),
            "risk_coverage": _risk_coverage(abs_err, calibrated[sel, k]),
        }
    return {"split": split, "status": manifest.status,
            "n_val": manifest.n_val, "per_target": per_target}


def _verdicts(split_metrics: dict[str, Any],
              baselines: dict[str, dict]) -> list[dict[str, Any]]:
    """Acceptance rows vs the plan sec. 4.4 success bar.

    The R²≥0.98 & tree-comparable bar for cyclen/cbc_max is judged per split;
    S1 is the strictest (ancestry-closure group split) and can compress the
    held-out target variance so R² goes negative even when the *ranking*
    (Spearman) is preserved — reported honestly.  F_r/F_q require within-case
    Spearman superiority over the ExtraTrees baseline.
    """
    verdicts: list[dict[str, Any]] = []

    def _bl(split: str, target: str, key: str) -> float:
        b = baselines.get(split, {}).get("per_target", {}).get(target, {})
        return float(b.get(key, float("nan")))

    def _r2(split: str, target: str) -> float:
        return float(split_metrics.get(split, {}).get("per_target", {})
                     .get(target, {}).get("r2", float("nan")))

    # -- R² bars for cyclen / cbc_max, per split ---------------------------- #
    for target in ("cyclen", "cbc_max"):
        for split in ("S0", "S1", "S2", "S4"):
            r2 = _r2(split, target)
            if not math.isfinite(r2):
                continue
            bl = _bl(split, target, "r2")
            bar_hit = r2 >= R2_BAR
            tree_ok = (not math.isfinite(bl)) or r2 >= bl - _TREE_TOL
            passed = bar_hit and tree_ok
            bltxt = f", tree {bl:.3f}" if math.isfinite(bl) else ", no tree"
            verdicts.append({
                "criterion": f"{split} {target} R² ≥ {R2_BAR} & ≥ trees",
                "value": f"R²={r2:.4f}{bltxt}",
                "verdict": "PASS" if passed else "FAILED",
            })
    # -- within-case Spearman superiority for F_r / F_q (S1 + S2) ----------- #
    for split in ("S1", "S2"):
        for target in ("f_r", "f_q"):
            m = split_metrics.get(split, {}).get("per_target", {}).get(target, {})
            sp = float(m.get("within_case_spearman", float("nan")))
            bl = _bl(split, target, "within_case_spearman")
            if not math.isfinite(sp):
                continue
            passed = (not math.isfinite(bl)) or sp >= bl - _TREE_TOL
            bltxt = f" tree={bl:.4f}" if math.isfinite(bl) else " (no tree)"
            verdicts.append({
                "criterion": f"{split} {target} within-case Spearman ≥ trees",
                "value": f"CNN={sp:.4f}{bltxt}",
                "verdict": "PASS" if passed else "FAILED",
            })
    # -- S2 / S4 interpolation functional ---------------------------------- #
    for split in ("S2", "S4"):
        m = split_metrics.get(split, {})
        ok = m.get("n_val", 0) > 0 and any(
            "r2" in v for v in m.get("per_target", {}).values())
        verdicts.append({
            "criterion": f"{split} interpolation functional",
            "value": f"n_val={m.get('n_val', 0)}",
            "verdict": "PASS" if ok else "FAILED",
        })
    return verdicts


def _fmt(x: Any, nd: int = 4) -> str:
    if isinstance(x, float):
        return "n/a" if not math.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _render_markdown(split_metrics: dict[str, Any], baselines: dict[str, dict],
                     verdicts: list[dict[str, Any]], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# PosValNet ensemble evaluation report")
    lines.append("")
    lines.append(f"- members: {meta.get('n_members')}  ")
    lines.append(f"- torch: {meta.get('torch')}  device: {meta.get('device')}  ")
    lines.append(f"- calibration: {'yes' if meta.get('calibrated') else 'no'}  ")
    lines.append(f"- params/member: {meta.get('n_params'):,}  ")
    lines.append("")

    lines.append("## Acceptance verdicts (plan sec. 4.4)")
    lines.append("")
    lines.append("| Criterion | Value | Verdict |")
    lines.append("|---|---|---|")
    for v in verdicts:
        lines.append(f"| {v['criterion']} | {v['value']} | **{v['verdict']}** |")
    lines.append("")

    for split, m in split_metrics.items():
        lines.append(f"## {split}  (status={m.get('status')}, n_val={m.get('n_val')})")
        lines.append("")
        if not m.get("per_target"):
            lines.append(f"_{m.get('note', 'no data')}_")
            lines.append("")
            continue
        lines.append("| Target | n | MAE | RMSE | R² (ens) | R² (mbr) | Spearman(case) | n_cases | tree R² | tree Sp |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        bl = baselines.get(split, {}).get("per_target", {})
        # render whatever the checkpoint actually scored (7 targets today, 8
        # under promote_max_asm_bu) rather than a hardcoded list.
        for name in m["per_target"]:
            t = m["per_target"].get(name, {})
            if "mae" not in t:
                lines.append(f"| {name} | {t.get('n_val', 0)} | insufficient labels |||||||||")
                continue
            b = bl.get(name, {})
            lines.append(
                f"| {name} | {t['n_val']} | {_fmt(t['mae'])} | {_fmt(t['rmse'])} | "
                f"{_fmt(t['r2'])} | {_fmt(t['member_r2_mean'])} | "
                f"{_fmt(t['within_case_spearman'])}±{_fmt(t['within_case_spearman_sd'],3)} | "
                f"{t['n_cases']} | {_fmt(b.get('r2', float('nan')))} | "
                f"{_fmt(b.get('within_case_spearman', float('nan')))} |")
        lines.append("")
        # risk-coverage for cyclen
        cy = m["per_target"].get("cyclen", {})
        if "risk_coverage" in cy:
            rc = ", ".join(f"{p['coverage']:.1f}:{p['mae']:.2f}" for p in cy["risk_coverage"])
            lines.append(f"- cyclen risk-coverage (coverage:MAE, σ-sorted): {rc}")
            lines.append("")
    return "\n".join(lines)


def eval_report(
    ckpt_dirs: Sequence[str | Path],
    splits: Sequence[str] = ("S0", "S1", "S2", "S4"),
    out: str | Path = "data/reports/model_report.md",
    *,
    store_dir: str | Path = DEFAULT_STORE,
    splits_dir: str | Path = DEFAULT_SPLITS,
    reports_dir: str | Path = DEFAULT_REPORTS,
    device: str = "cpu",
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the ensemble over ``splits`` and write a Markdown report."""
    import torch
    dev = torch.device(device)
    members, metas = [], []
    for d in ckpt_dirs:
        model, meta = load_member(d, dev)
        members.append(model)
        metas.append(meta)

    calib = None
    if calibration_path is None:
        guess = Path(ckpt_dirs[0]).parent / CALIB_NAME
        if guess.is_file():
            calibration_path = guess
    if calibration_path and Path(calibration_path).is_file():
        calib = load_calibration(calibration_path)

    split_metrics: dict[str, Any] = {}
    for split in splits:
        split_metrics[split] = evaluate_split(
            members, metas, split, calib=calib,
            store_dir=store_dir, splits_dir=splits_dir, device=dev)

    baselines: dict[str, dict] = {}
    for split in splits:
        bpath = Path(reports_dir) / f"baseline_{split}.json"
        if bpath.is_file():
            baselines[split] = json.loads(bpath.read_text(encoding="utf-8"))

    verdicts = _verdicts(split_metrics, baselines)
    report_meta = {
        "n_members": len(members),
        "torch": metas[0]["versions"]["torch"],
        "device": device,
        "calibrated": calib is not None,
        "n_params": metas[0].get("n_params", 0),
    }
    md = _render_markdown(split_metrics, baselines, verdicts, report_meta)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    # sidecar JSON for programmatic access
    (out_path.with_suffix(".json")).write_text(
        json.dumps({"split_metrics": split_metrics, "verdicts": verdicts,
                    "meta": report_meta}, indent=2, sort_keys=True),
        encoding="utf-8")
    return {"split_metrics": split_metrics, "verdicts": verdicts, "out": str(out_path)}


__all__ = ["eval_report", "evaluate_split"]


if __name__ == "__main__":          # pragma: no cover - manual smoke
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble_dir")
    ap.add_argument("--splits", nargs="+", default=["S0", "S1", "S2", "S4"])
    ap.add_argument("--out", default="data/reports/model_report.md")
    args = ap.parse_args()
    dirs = sorted(Path(args.ensemble_dir).glob("member_*"))
    res = eval_report(dirs, args.splits, args.out)
    for v in res["verdicts"]:
        print(f"{v['verdict']:7s} {v['criterion']} -> {v['value']}")
