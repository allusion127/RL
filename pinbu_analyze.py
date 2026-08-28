"""Readout for the measured pin-burnup wave (``pinbu_wave.py``).

Scores ONLY the rules pre-registered in
``data/reports/pinbu_wave_prereg_20260820.md`` sec. 6.  Nothing here chooses a
threshold or a hypothesis after the fact:

  1. delivery verdict  = measured ``max_pin_burnup`` <= 80.0 GWd/tU (LEU+ limit),
     per core, with the deck's own 78.0 acquisition gate reported alongside;
  2. pin-head accuracy = bias mean(pred - meas) and MAE, per feed and pooled,
     with a bootstrap 95 % CI on the bias;
  3. H1 (head biased ~ +9) vs H2 (pool deficit, bias ~ -1 +- 2) decided from that
     CI by the pre-registered rule;
  4. recalibration recommendation triggered at |pooled bias| > 2.0 GWd/tU (the
     margin the decks already spend: minfr_pin_bu_limit 78 == 80 - 2).

A chain that failed any registered control (non-converged, determinism, restart
provenance) contributes to NO statistic and gets NO verdict -- it is listed under
run integrity and dropped.  The out-of-support comparison group (the 157 pin
labels the store already held at four of these cells, all at F_r >= 1.70) is
re-scored with the same champion so the wave's in-operating-region bias can be
read against the in-support bias the prereg registered as its prior.

    python pinbu_analyze.py --results runs/pinbu_wave/pinbu_wave_results.jsonl

Writes ``data/reports/pinbu_wave_results_20260820.{md,json}``.  Read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from pinbu_wave import (                                       # noqa: E402
    ACQUISITION_PIN_LIMIT, LICENSING_PIN_LIMIT, CALIB_CELLS)

#: prereg sec. 6.4 -- the recalibration trigger, == the margin the decks spend.
RECAL_TRIGGER_GWD = 2.0
#: prereg sec. 5 -- the two registered hypotheses, in bias (pred - meas) units.
H1_BIAS, H2_BIAS, H2_HALFWIDTH = 9.0, -1.0, 2.0
BOOT_REPS, BOOT_SEED = 10000, 0


def _boot_ci(x, reps: int = BOOT_REPS, seed: int = BOOT_SEED):
    import numpy as np
    a = np.asarray([v for v in x if v is not None and math.isfinite(v)], float)
    if a.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(reps, a.size))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _stats(errors) -> dict:
    import numpy as np
    a = np.asarray([e for e in errors if e is not None and math.isfinite(e)], float)
    if a.size == 0:
        return {"n": 0}
    lo, hi = _boot_ci(a)
    return {"n": int(a.size), "bias": float(a.mean()), "mae": float(np.abs(a).mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()),
            "ci95_lo": lo, "ci95_hi": hi}


def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "n=0"
    return (f"n={s['n']:2d}  bias {s['bias']:+6.2f}  MAE {s['mae']:5.2f}  "
            f"sd {s['sd']:5.2f}  95% CI [{s['ci95_lo']:+.2f}, {s['ci95_hi']:+.2f}]")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pinbu_analyze.py")
    ap.add_argument("--results", default="runs/pinbu_wave/pinbu_wave_results.jsonl")
    ap.add_argument("--plan", default="data/reports/pinbu_wave_prereg_20260820.json")
    ap.add_argument("--store-dir", default="data/store")
    ap.add_argument("--model", default="data/models/s1i")
    ap.add_argument("--out", default="data/reports/pinbu_wave_results_20260820")
    ap.add_argument("--no-support-group", action="store_true",
                    help="skip re-scoring the 157 pre-existing measured labels")
    ap.add_argument("--support-only", action="store_true",
                    help=argparse.SUPPRESS)   # internal: the subprocess entry
    args = ap.parse_args(argv)

    # The support group needs torch.  Importing torch AFTER pandas/pyarrow have
    # been pulled in by this process fails on this box with a shm.dll load error
    # (and importing it first collides with numpy's MKL OpenMP), so the scoring
    # runs in its own interpreter where torch is the first heavy import.
    if args.support_only:
        print(json.dumps(_support_group(args, None)))
        return 0

    import numpy as np

    res_path = Path(args.results) if Path(args.results).is_absolute() else BASE / args.results
    plan = json.loads((BASE / args.plan if not Path(args.plan).is_absolute()
                       else Path(args.plan)).read_text(encoding="utf-8"))
    rows = [json.loads(ln) for ln in res_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]

    # -- 1. run integrity --------------------------------------------------- #
    integrity = {
        "n_planned": int(plan["n_targets"]),
        "n_chains": len(rows),
        "converged": sum(r["status"] == "converged" for r in rows),
        "nonconverged": sum(r["status"] == "nonconverged" for r in rows),
        "error": sum(r["status"] == "error" for r in rows),
        "determinism_ok": sum(bool(r["determinism_ok"]) for r in rows),
        "provenance_ok": sum(bool(r["provenance_ok"]) for r in rows),
        "pin_present": sum(r["measured"]["max_pin_burnup"] is not None for r in rows),
    }
    good, dropped = [], []
    for r in rows:
        if (r["status"] == "converged" and r["determinism_ok"] and r["provenance_ok"]
                and r["measured"]["max_pin_burnup"] is not None):
            good.append(r)
        else:
            why = []
            if r["status"] != "converged":
                why.append(f"status={r['status']}" + (f" ({r['failure']})" if r["failure"] else ""))
            if not r["determinism_ok"]:
                d = r["deltas_measured_minus_stored"]
                # An error chain has no deltas at all; saying "determinism: "
                # with nothing after it reads as a second, separate failure.
                shown = ", ".join(f"d{k}={v:+.4g}" for k, v in d.items()
                                  if v is not None and abs(v) > 0)
                if shown:
                    why.append("determinism: " + shown)
            if not r["provenance_ok"]:
                why.append(f"provenance {r['planned_provenance']} -> {r['restart_provenance']}")
            if r["measured"]["max_pin_burnup"] is None:
                why.append("no PPI pin value")
            dropped.append({"record_id": r["record_id"], "group": r["group"],
                            "why": "; ".join(why)})
    integrity["usable"] = len(good)

    # Determinism magnitudes on the chains that DID converge -- the free control.
    det = {k: _stats([r["deltas_measured_minus_stored"][k] for r in rows
                      if r["status"] == "converged"])
           for k in ("f_r", "cyclen", "cbc_max", "f_q", "ao_abs",
                     "max_assembly_burnup")}

    # -- 2. delivery verdicts ------------------------------------------------ #
    delivery: dict[str, dict] = {}
    for g in [m for m in plan["groups"] if m["kind"] == "delivery"]:
        members = [r for r in good if r["group"] == g["name"]]
        planned = [t for t in plan["targets"] if t["group"] == g["name"]]
        cores = []
        for t in sorted(planned, key=lambda t: t["rank_in_group"]):
            r = next((x for x in members if x["record_id"] == t["record_id"]), None)
            cores.append({
                "record_id": t["record_id"], "rank": t["rank_in_group"],
                "stored_f_r": t["stored"]["f_r"],
                "predicted_pin": t["predicted"]["max_pin_burnup"],
                "measured_pin": None if r is None else r["measured"]["max_pin_burnup"],
                "error": None if r is None else r["pin_error_pred_minus_meas"],
                "verdict_80": None if r is None else r["delivery_verdict"],
                "under_78": None if r is None
                else bool(r["measured"]["max_pin_burnup"] <= ACQUISITION_PIN_LIMIT),
            })
        got = [c for c in cores if c["measured_pin"] is not None]
        delivery[g["name"]] = {
            "case": f"{g.get('campaign','')}",
            "n_measured": len(got),
            "n_pass_80": sum(c["verdict_80"] == "PASS" for c in got),
            "n_pass_78": sum(bool(c["under_78"]) for c in got),
            "winner_verdict": (cores[0]["verdict_80"] if cores else None),
            "winner_measured_pin": (cores[0]["measured_pin"] if cores else None),
            "best_measured_pin": (min(c["measured_pin"] for c in got) if got else None),
            "predicted_verdict_80": (
                "PASS" if all(c["predicted_pin"] <= LICENSING_PIN_LIMIT for c in cores)
                else ("FAIL" if all(c["predicted_pin"] > LICENSING_PIN_LIMIT for c in cores)
                      else "MIXED")),
            "cores": cores,
        }

    # -- 3. pin-head accuracy ------------------------------------------------ #
    errs = [r["pin_error_pred_minus_meas"] for r in good]
    by_feed = {str(f): _stats([r["pin_error_pred_minus_meas"] for r in good
                               if r["feed"] == f])
               for f in sorted({r["feed"] for r in good})}
    by_role = {role: _stats([r["pin_error_pred_minus_meas"] for r in good
                             if r["role"] == role])
               for role in ("delivery", "calibration")}
    by_group = {g: _stats([r["pin_error_pred_minus_meas"] for r in good
                           if r["group"] == g])
                for g in sorted({r["group"] for r in good})}
    pooled = _stats(errs)
    calib_only = _stats([r["pin_error_pred_minus_meas"] for r in good
                         if r["role"] == "calibration"])

    # -- 4. H1 vs H2 (prereg sec. 6.3), on the CALIBRATION set --------------- #
    lo, hi = calib_only.get("ci95_lo", float("nan")), calib_only.get("ci95_hi", float("nan"))
    covers_h1 = bool(lo <= H1_BIAS <= hi) if math.isfinite(lo) else False
    covers_h2 = bool(lo <= H2_BIAS + H2_HALFWIDTH and hi >= H2_BIAS - H2_HALFWIDTH) \
        if math.isfinite(lo) else False
    if covers_h1 and covers_h2:
        verdict, why = "UNDERPOWERED", ("the 95% CI spans both registered "
                                        "hypotheses; neither is claimed")
    elif covers_h1:
        verdict, why = "H1", ("the head over-predicts pin by ~9 GWd/tU on these "
                              "cores; recalibrate before any low-feed verdict")
    elif covers_h2:
        verdict, why = "H2", ("the head is ~unbiased here; the low-feed map "
                              "closure is a search/design result, and mesh_multitype "
                              "5.1's pessimism claim is a cross-core artefact")
    else:
        verdict, why = "NEITHER", ("the measured bias falls outside both "
                                   "registered hypotheses; report as an open finding")

    # -- 5. calibration curve ------------------------------------------------ #
    curve = {}
    cal = [r for r in good if r["role"] == "calibration"]
    if len(cal) >= 3:
        x = np.asarray([r["predicted"]["max_pin_burnup"] for r in cal], float)
        y = np.asarray([r["measured"]["max_pin_burnup"] for r in cal], float)
        slope, icept = np.polyfit(x, y, 1)
        resid = y - (slope * x + icept)
        curve = {"n": len(cal), "slope": float(slope), "intercept": float(icept),
                 "resid_sd": float(resid.std(ddof=2)) if len(cal) > 2 else None,
                 "pred_span": [float(x.min()), float(x.max())],
                 "meas_span": [float(y.min()), float(y.max())],
                 "r": float(np.corrcoef(x, y)[0, 1]) if len(cal) > 2 else None,
                 "spearman_rank_pairs": len(cal)}

    # -- 6. recalibration recommendation (prereg sec. 6.4) ------------------- #
    bias = pooled.get("bias", float("nan"))
    trigger = bool(math.isfinite(bias) and abs(bias) > RECAL_TRIGGER_GWD)
    recal = {
        "pooled_bias": bias, "trigger_gwd": RECAL_TRIGGER_GWD,
        "triggered": trigger,
        "recommendation": (
            "FIT pinbu_physics for the next champion: "
            "lpopt/model/pinbu_physics.py::fit_pinbu_physics(model_dir, store_dir, "
            "splits_dir, library_id=...) -- no champion s1c..s1i ships a "
            "pinbu_physics.json, so the served pin column is the raw head today. "
            "Fit ga80 and paramA separately (the function takes ONE library_id)."
            if trigger else
            "LEAVE THE SERVE PATH ALONE. The measured |bias| is inside the 2.0 "
            "GWd/tU margin the decks already spend (minfr_pin_bu_limit 78 = 80-2); "
            "record the measured MAE as that margin's empirical basis."),
    }

    # -- 7. out-of-support comparison group (the prereg's registered prior) -- #
    support = {}
    if not args.no_support_group:
        import subprocess
        cmd = [sys.executable, str(Path(__file__).resolve()), "--support-only",
               "--results", str(res_path), "--store-dir", args.store_dir,
               "--model", args.model]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE),
                               timeout=1800)
            support = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as exc:                            # noqa: BLE001
            tail = (p.stderr[-400:] if "p" in dir() and p.stderr else "")
            support = {"error": f"{type(exc).__name__}: {exc}", "stderr_tail": tail}

    out = {
        "schema": "pinbu_wave_results_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prereg": "data/reports/pinbu_wave_prereg_20260820.md",
        "results_file": str(res_path),
        "integrity": integrity, "dropped": dropped,
        "determinism": det,
        "delivery": delivery,
        "accuracy": {"pooled": pooled, "by_feed": by_feed, "by_role": by_role,
                     "by_group": by_group},
        "hypothesis": {"verdict": verdict, "why": why,
                       "calibration_ci95": [lo, hi],
                       "H1_bias": H1_BIAS, "H2_bias": H2_BIAS,
                       "H2_halfwidth": H2_HALFWIDTH},
        "calibration_curve": curve,
        "recalibration": recal,
        "out_of_support_prior": support,
    }
    stem = Path(args.out) if Path(args.out).is_absolute() else BASE / args.out
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n",
                                         encoding="utf-8")
    _render(out, stem.with_suffix(".md"))
    _print(out)
    print(f"\n-> {stem.with_suffix('.md')}\n-> {stem.with_suffix('.json')}")
    return 0


def _support_group(args, plan) -> dict:
    """Re-score the pin labels the store ALREADY held at these cells.

    Registered in the prereg sec. 5 as the prior: 157 rows at four of the eight
    calibration cells, every one at F_r >= 1.70 (inside the training support and
    outside the operating region).  Re-scored HERE with the same champion so the
    wave's in-operating-region bias is read against a number produced by the same
    code path, not against a transcription.
    """
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(8)
    from lpopt.data.schema import unpack_pattern
    from lpopt.model.model_api import PosValCnnBackend
    from lpopt.model.model_api import _TARGET_TO_SURROGATE_COL as COL
    from lpopt.vendor.masterrl.domain import CaseKey

    store_dir = BASE / args.store_dir if not Path(args.store_dir).is_absolute() \
        else Path(args.store_dir)
    df = pd.read_parquet(store_dir / "records.parquet")
    be = PosValCnnBackend.from_dir(
        BASE / args.model if not Path(args.model).is_absolute() else Path(args.model),
        store_dir=store_dir, library_id="ga80", device="cpu")

    cells = {(c["case_pair"], int(c["feed"])) for c in CALIB_CELLS}
    out = {}
    all_err: list[float] = []
    for pair, feed in sorted(cells):
        s = df[(df.case_pair == pair) & (df.feed == feed) & df.max_pin_burnup.notna()]
        if s.empty:
            continue
        ec = float(s.e_core.median())
        pats = [unpack_pattern(str(p)) for p in s["pattern"]]
        pred = be.predict(pats, CaseKey(pair, feed), ec)
        err = (np.asarray(pred.mean[:, COL["max_pin_burnup"]], float)
               - s.max_pin_burnup.to_numpy(float))
        out[f"{pair}/f{feed}"] = {
            **_stats(list(err)),
            "f_r_min": float(s.f_r.min()),
            "meas_span": [float(s.max_pin_burnup.min()), float(s.max_pin_burnup.max())],
        }
        all_err.extend(err.tolist())
    out["POOLED"] = _stats(all_err)
    return out


def _print(o: dict) -> None:
    i = o["integrity"]
    print(f"\n{'='*78}\nRUN INTEGRITY  {i['n_chains']}/{i['n_planned']} chains  "
          f"converged {i['converged']}  determinism_ok {i['determinism_ok']}  "
          f"provenance_ok {i['provenance_ok']}  USABLE {i['usable']}")
    for d in o["dropped"]:
        print(f"  [dropped] {d['record_id'][:12]} {d['group']:20s} {d['why']}")
    print(f"\nDETERMINISM (converged chains, measured - stored)")
    for k, s in o["determinism"].items():
        if s.get("n"):
            print(f"  {k:20s} n={s['n']:2d} max|d| "
                  f"{max(abs(s['min']), abs(s['max'])):.4g}")
    print(f"\nDELIVERY VERDICTS (limit {LICENSING_PIN_LIMIT} GWd/tU)")
    for name, g in o["delivery"].items():
        wm = "--" if g["winner_measured_pin"] is None else f"{g['winner_measured_pin']:.2f}"
        bm = "--" if g["best_measured_pin"] is None else f"{g['best_measured_pin']:.2f}"
        print(f"  {name:22s} predicted {g['predicted_verdict_80']:5s} -> MEASURED "
              f"{g['n_pass_80']}/{g['n_measured']} PASS   winner "
              f"{g['winner_verdict']} at {wm}   best {bm}")
    print(f"\nPIN-HEAD ACCURACY (pred - measured, GWd/tU)")
    print(f"  {'POOLED':22s} {_fmt(o['accuracy']['pooled'])}")
    for k, s in o["accuracy"]["by_feed"].items():
        print(f"  feed {k:17s} {_fmt(s)}")
    for k, s in o["accuracy"]["by_role"].items():
        print(f"  {k:22s} {_fmt(s)}")
    sup = o.get("out_of_support_prior") or {}
    if "POOLED" in sup:
        print(f"  {'(prior: in-support)':22s} {_fmt(sup['POOLED'])}")
    h = o["hypothesis"]
    print(f"\nHYPOTHESIS  ->  {h['verdict']}\n  {h['why']}")
    c = o["calibration_curve"]
    if c:
        print(f"\nCALIBRATION CURVE  measured = {c['slope']:.4f} * predicted "
              f"{c['intercept']:+.3f}   (n={c['n']}, r={c['r']:.3f}, "
              f"resid sd {c['resid_sd']:.2f}) over predicted "
              f"{c['pred_span'][0]:.1f}-{c['pred_span'][1]:.1f}")
    r = o["recalibration"]
    print(f"\nRECALIBRATION  triggered={r['triggered']} "
          f"(|bias| {abs(r['pooled_bias']):.2f} vs {r['trigger_gwd']})\n  "
          f"{r['recommendation']}")


def _render(o: dict, path: Path) -> None:
    L: list[str] = []
    a = L.append
    i = o["integrity"]
    a("# Measured pin-burnup wave — RESULTS (2026-08-20)\n")
    a(f"Scored strictly against the pre-registered rules in `{o['prereg']}` §6. "
      f"Machine-readable twin: `{Path(o['prereg']).stem.replace('prereg','results')}.json`.\n")
    a("## 1. Run integrity\n")
    a(f"| chains | converged | determinism ok | provenance ok | pin present | **usable** |")
    a("|---:|---:|---:|---:|---:|---:|")
    a(f"| {i['n_chains']}/{i['n_planned']} | {i['converged']} | {i['determinism_ok']} "
      f"| {i['provenance_ok']} | {i['pin_present']} | **{i['usable']}** |\n")
    if o["dropped"]:
        a("Dropped (no verdict, no statistic — a failed control is a refusal, "
          "never a downgraded result):\n")
        for d in o["dropped"]:
            a(f"* `{d['record_id'][:12]}` {d['group']} — {d['why']}")
        a("")
    else:
        a("No chain failed a registered control.\n")
    a("Determinism control (measured − stored, converged chains):\n")
    a("| axis | n | max abs delta | tolerance |")
    a("|---|---:|---:|---:|")
    tol = {"f_r": "0.002", "cyclen": "0.5", "cbc_max": "2.0"}
    for k, s in o["determinism"].items():
        if s.get("n"):
            a(f"| {k} | {s['n']} | {max(abs(s['min']), abs(s['max'])):.4g} "
              f"| {tol.get(k, '—')} |")
    a("")
    a(f"## 2. Delivery verdicts (LEU+ limit {LICENSING_PIN_LIMIT} GWd/tU)\n")
    a("| group | predicted | **measured PASS/n** | winner measured pin | winner verdict | best measured pin | ≤78 |")
    a("|---|---|---:|---:|---|---:|---:|")
    for name, g in o["delivery"].items():
        wm = "—" if g["winner_measured_pin"] is None else f"{g['winner_measured_pin']:.2f}"
        bm = "—" if g["best_measured_pin"] is None else f"{g['best_measured_pin']:.2f}"
        a(f"| `{name}` | {g['predicted_verdict_80']} | **{g['n_pass_80']}/{g['n_measured']}** "
          f"| {wm} | **{g['winner_verdict']}** | {bm} | {g['n_pass_78']}/{g['n_measured']} |")
    a("")
    for name, g in o["delivery"].items():
        a(f"### {name}\n")
        a("| rank | record | stored F_r | predicted pin | **measured pin** | pred − meas | verdict |")
        a("|---:|---|---:|---:|---:|---:|---|")
        for c in g["cores"]:
            mm = "—" if c["measured_pin"] is None else f"**{c['measured_pin']:.3f}**"
            ee = "—" if c["error"] is None else f"{c['error']:+.2f}"
            a(f"| {c['rank']} | `{c['record_id'][:12]}` | {c['stored_f_r']:.4f} "
              f"| {c['predicted_pin']:.2f} | {mm} | {ee} | {c['verdict_80'] or '—'} |")
        a("")
    a("## 3. Pin-head accuracy (predicted − measured, GWd/tU)\n")
    a("| slice | n | bias | MAE | sd | 95% CI on bias |")
    a("|---|---:|---:|---:|---:|---|")

    def _row(label, s):
        if not s.get("n"):
            return
        a(f"| {label} | {s['n']} | {s['bias']:+.2f} | {s['mae']:.2f} | {s['sd']:.2f} "
          f"| [{s['ci95_lo']:+.2f}, {s['ci95_hi']:+.2f}] |")

    _row("**POOLED**", o["accuracy"]["pooled"])
    for k, s in o["accuracy"]["by_feed"].items():
        _row(f"feed {k}", s)
    for k, s in o["accuracy"]["by_role"].items():
        _row(k, s)
    for k, s in o["accuracy"]["by_group"].items():
        _row(f"`{k}`", s)
    sup = o.get("out_of_support_prior") or {}
    if "POOLED" in sup:
        a("")
        a("Registered prior — the labels the store already held at these cells "
          "(all F_r ≥ 1.70, i.e. inside the training support and outside the "
          "operating region), re-scored with the same champion:\n")
        a("| cell | n | bias | MAE | measured span | min F_r |")
        a("|---|---:|---:|---:|---|---:|")
        for k, s in sup.items():
            if k == "POOLED" or not s.get("n"):
                continue
            a(f"| {k} | {s['n']} | {s['bias']:+.2f} | {s['mae']:.2f} "
              f"| {s['meas_span'][0]:.1f}–{s['meas_span'][1]:.1f} | {s['f_r_min']:.2f} |")
        p = sup["POOLED"]
        a(f"| **pooled prior** | {p['n']} | **{p['bias']:+.2f}** | {p['mae']:.2f} | | |")
    a("")
    h = o["hypothesis"]
    a(f"## 4. Registered hypothesis test → **{h['verdict']}**\n")
    a(f"Calibration-set bias 95% CI: **[{h['calibration_ci95'][0]:+.2f}, "
      f"{h['calibration_ci95'][1]:+.2f}]** against H1 = {h['H1_bias']:+.0f} "
      f"(head bias) and H2 = {h['H2_bias']:+.0f} ± {h['H2_halfwidth']:.0f} "
      f"(pool deficit).\n")
    a(f"{h['why']}\n")
    c = o["calibration_curve"]
    if c:
        a("## 5. Calibration curve\n")
        a(f"`measured = {c['slope']:.4f} × predicted {c['intercept']:+.3f}` "
          f"(n={c['n']}, r={c['r']:.3f}, residual sd {c['resid_sd']:.2f}), fitted "
          f"over predicted {c['pred_span'][0]:.1f}–{c['pred_span'][1]:.1f} → "
          f"measured {c['meas_span'][0]:.1f}–{c['meas_span'][1]:.1f}. Outside that "
          f"predicted span the curve is not claimed.\n")
    r = o["recalibration"]
    a("## 6. Recalibration recommendation\n")
    a(f"Pooled bias **{r['pooled_bias']:+.2f}** GWd/tU against the "
      f"{r['trigger_gwd']:.1f} trigger → **{'TRIGGERED' if r['triggered'] else 'NOT triggered'}**.\n")
    a(f"{r['recommendation']}\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
