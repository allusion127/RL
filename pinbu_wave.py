"""Measured pin-burnup re-evaluation of STORED cores (the pin-BU measurement path).

WHY.  The programme has ZERO measured pin burnup on its own campaign cores.
``max_pin_burnup`` is only written when the equilibrium runner is built with
``enable_pin_burnup=True``, which turns on the ``%EDT_OPT ipin=1`` PPI edit
(``vendor/masterrl/equilibrium.py:461-466`` -> ``burnup.enable_ppi_output``).
That flag is reachable from exactly two places:

  * ``[design].enable_pin_burnup``  -> ``design/pathfinder.py:222`` (bootstrap only)
  * ``curriculum.make_pin_burnup_verifier`` (curriculum.py:1017) -> a WaveVerifier
    whose injected evaluator factory sets ``enable_pin_burnup=True``

``lpopt optimize`` uses neither: ``WaveVerifier._default_factory`` hard-codes
``enable_pin_burnup=False`` (``search/verify.py:851``).  Every fpcamp row in the
store therefore has ``max_pin_burnup`` null (2,270 rows, 0 non-null), so both

  (a) the DELIVERY verdicts of the opened-cell winners against the LEU+ 80
      GWd/tU pin limit, and
  (b) the low-feed PIN-HEAD calibration (the DB comparison measured our head
      ~9 GWd/tU pessimistic at f109/f113 -- mesh_multitype README 5.1)

rest on a PREDICTED pin number from a head that ``data/reports/pinbu_forensics.md``
shows has ~0 within-cell held-out rank skill.

WHAT.  This module is the missing measurement path: a FIXED-PATTERN
re-evaluation harness.  It takes SPECIFIED store records, replays each one's own
stored pattern through the SAME asset resolution the campaign used, but with
``make_pin_burnup_verifier`` instead of the default verifier -- so the identical
chain runs and additionally emits the MAS_PPI 3-D pin burnup.

Nothing here reinvents evaluation.  ``fr_arms.py`` is the fixed-pattern
precedent (store pattern -> ``WaveEntry`` -> ``WaveVerifier.evaluate_wave``);
``curriculum.make_pin_burnup_verifier`` is the pin-burnup precedent.  This module
only (a) selects records, (b) pins the model's prediction BEFORE the run,
(c) wires the two precedents together, and (d) patches the measured column into
the canonical store.

DETERMINISM IS THE CONTROL.  MASTER is deterministic on this fleet (the
``fr_arms`` A0 control reproduced F_r to 0.0000), the pattern is the record's
own, and ``%EDT_OPT ipin`` is an EDIT flag -- output only.  So every chain must
reproduce its stored ``f_r`` / ``cyclen`` / ``cbc_max`` labels to
:data:`TOL_F_R` / :data:`TOL_CYCLEN` / :data:`TOL_CBC`.  A chain that does NOT is
reported ``determinism_ok=False`` and its pin value is NOT merged: it means the
re-run was not the same evaluation, so its pin number does not belong to the
stored core.  One chain per core is enough precisely because of this.

USAGE

    # 1. PLAN (local, read-only, no MASTER).  Selects the target set and PINS the
    #    champion's predicted pin burnup for every core BEFORE anything runs.
    python pinbu_wave.py plan --model data/models/s1i --out data/reports/pinbu_wave_prereg_20260820.json

    # 2. RUN (on the MASTER box; one chain per core)
    python pinbu_wave.py run --plan data/reports/pinbu_wave_prereg_20260820.json \
        --deck pinbu_wave_199.inp --run-dir runs/pinbu_wave

    # 3. PATCH the measured column into the canonical store (in place, back up first)
    python pinbu_wave.py patch --results runs/pinbu_wave/pinbu_wave_results.jsonl --dry-run
    python pinbu_wave.py patch --results runs/pinbu_wave/pinbu_wave_results.jsonl

``plan`` and ``patch --dry-run`` are read-only on ``data/store``.  Readouts live
in ``pinbu_analyze.py``.
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

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: LEU+ licensing peak-pin discharge burnup limit; the DELIVERY verdict bar.
LICENSING_PIN_LIMIT = 80.0
#: The acquisition gate the campaigns actually searched under (80 - 2.0 model
#: margin, ``[acquisition].minfr_pin_bu_limit``).  Reported, not a verdict.
ACQUISITION_PIN_LIMIT = 78.0

#: Determinism gate: stored label vs re-run.  Same numbers ``regen_chain.py``
#: uses, except F_r which is tightened to the F3 resolution floor -- this replay
#: is restart-seeded from the SAME restart, not bootstrapped from cy1, so it has
#: no equilibrium-non-uniqueness slack to spend.
TOL_F_R, TOL_CYCLEN, TOL_CBC = 2.0e-3, 0.5, 2.0

RESULTS_NAME = "pinbu_wave_results.jsonl"
PLAN_SCHEMA = "pinbu_wave_prereg_v1"

#: The four DELIVERY sets: the opened-cell winners whose verdicts are currently
#: predicted-only.  ``f_r_limit`` is the deck's own ``[acquisition].f_r_limit``
#: where the campaign could reach it; the two paramA cells never got below 1.55
#: (best 1.6088 / 1.5956 raw), so their feasibility is the joint CBC/F_q/AO
#: gate set and F_r is the ranking objective -- exactly how their result
#: reports rank "joint-clean" cores.
DELIVERY_GROUPS = [
    {"name": "N1N2_f113", "campaign": "fpcamp_minfr_N1N2_f113",
     "library_id": "ga80", "top_n": 5, "f_r_limit": 1.55},
    {"name": "E1E2_f109", "campaign": "fpcamp_minfr_E1E2_f109",
     "library_id": "ga80", "top_n": 5, "f_r_limit": 1.55},
    {"name": "HGD569_f125_2type", "campaign": "fpcamp_minfr_hgd569_f125",
     "library_id": "paramA", "top_n": 5, "f_r_limit": None},
    {"name": "HGD569_f125_3type", "campaign": "fpcamp_minfr_triple_f125",
     "library_id": "paramA", "top_n": 5, "f_r_limit": None},
    {"name": "HGD569_f125_3type_r2", "campaign": "fpcamp_minfr_triple_f125_r2",
     "library_id": "paramA", "top_n": 5, "f_r_limit": None},
]

#: Shared joint-feasibility gates (every deck sets the same three).
CBC_LIMIT, F_Q_LIMIT, AO_ABS_LIMIT = 1600.0, 2.41, 0.30

#: CALIBRATION set: the seven cells the DB comparison flagged as pin-pessimistic
#: (mesh_multitype README 5.1 -- f109 e5.2/5.3/5.4/5.5 and f113 e5.3/5.4/5.5),
#: plus E1_E2/f109 (the e5.0 delivery cell, which supplies the only large
#: low-predicted-pin population in the low-feed band).
#:
#: Cores are drawn to SPAN the predicted-pin axis -- a calibration CURVE needs
#: spread, not the best five cores -- so each cell is cut into ``bins``
#: equal-width predicted-pin bins over its OWN ``[pin_lo, pin_hi]`` window and
#: the core nearest each bin centre is taken.  The windows below are pinned from
#: a read-only s1i scan of the store (they are that cell's actual predicted-pin
#: support; a fixed 74-84 window would miss five of the eight cells entirely,
#: whose store cores start at 79-83).
#:
#: BUDGET ASYMMETRY IS DELIBERATE.  ``f113`` carries 13 of the 24 slots because
#: the store holds ZERO measured pin burnup at feed 113 -- any chain here is the
#: first measurement at that feed.  The three f109 cells that already hold 34-50
#: measured labels get ONE core each, placed at the LOW-predicted-pin end their
#: existing labels do not reach.
CALIB_CELLS: list[dict] = [
    # -- feed 113: no measured pin exists anywhere in the store ------------- #
    {"name": "calib_N1N2_f113", "case_pair": "N1_N2", "feed": 113,
     "library_id": "ga80", "pin_lo": 78.0, "pin_hi": 94.0, "bins": 7, "per_bin": 1},
    {"name": "calib_L1L2_f113", "case_pair": "L1_L2", "feed": 113,
     "library_id": "ga80", "pin_lo": 79.0, "pin_hi": 92.0, "bins": 3, "per_bin": 1},
    {"name": "calib_G3G4_f113", "case_pair": "G3_G4", "feed": 113,
     "library_id": "ga80", "pin_lo": 80.0, "pin_hi": 96.0, "bins": 3, "per_bin": 1},
    # -- feed 109: measured, but only in the high-F_r (unoptimized) region --- #
    {"name": "calib_E1E2_f109", "case_pair": "E1_E2", "feed": 109,
     "library_id": "ga80", "pin_lo": 76.0, "pin_hi": 88.0, "bins": 5, "per_bin": 1},
    {"name": "calib_K1K2_f109", "case_pair": "K1_K2", "feed": 109,
     "library_id": "ga80", "pin_lo": 79.0, "pin_hi": 92.0, "bins": 3, "per_bin": 1},
    {"name": "calib_L1L2_f109", "case_pair": "L1_L2", "feed": 109,
     "library_id": "ga80", "pin_lo": 74.0, "pin_hi": 80.0, "bins": 1, "per_bin": 1},
    {"name": "calib_N1N2_f109", "case_pair": "N1_N2", "feed": 109,
     "library_id": "ga80", "pin_lo": 83.0, "pin_hi": 86.0, "bins": 1, "per_bin": 1},
    {"name": "calib_G3G4_f109", "case_pair": "G3_G4", "feed": 109,
     "library_id": "ga80", "pin_lo": 82.0, "pin_hi": 86.0, "bins": 1, "per_bin": 1},
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _finite(value) -> bool:
    v = _f(value)
    return bool(v == v and math.isfinite(v))


def joint_clean(row, f_r_limit: float | None) -> bool:
    """The gate set every campaign deck shares, plus the optional F_r limit."""
    if not (bool(row.get("valid")) and bool(row.get("converged"))):
        return False
    for name, limit in (("cbc_max", CBC_LIMIT), ("f_q", F_Q_LIMIT),
                        ("ao_abs", AO_ABS_LIMIT)):
        v = _f(row.get(name))
        if not _finite(v) or v > limit:
            return False
    fr = _f(row.get("f_r"))
    if not _finite(fr):
        return False
    if f_r_limit is not None and fr > f_r_limit:
        return False
    return True


def load_store(store_dir: Path):
    import pandas as pd
    return pd.read_parquet(Path(store_dir) / "records.parquet")


# --------------------------------------------------------------------------- #
# PLAN — select the target set and pin the prediction BEFORE the spend
# --------------------------------------------------------------------------- #
def _score(backend, rows, library_id: str) -> list[dict]:
    """Champion prediction for each store row, on the SERVE path the campaign used.

    ``predict(patterns, CaseKey, cell)`` is what acquisition gates on, so that --
    not ``predict_rows_raw`` -- is what a delivery verdict must be compared
    against.  Every surrogate column is pinned, not just the pin axis: the other
    five are the free determinism/skill controls this wave gets at zero cost.
    """
    import numpy as np
    from lpopt.data.schema import unpack_pattern
    from lpopt.model.model_api import _TARGET_TO_SURROGATE_COL as COL
    from lpopt.vendor.masterrl.domain import CaseKey

    names = ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs", "max_pin_burnup")
    out: list[dict] = []
    # One predict() call per (pair, feed, e_core) so the case/cell handed to the
    # backend is the row's own, never a batch-broadcast neighbour's.
    groups: dict[tuple, list] = {}
    for _, row in rows.iterrows():
        key = (str(row["case_pair"]), int(row["feed"]), round(_f(row["e_core"]), 6))
        groups.setdefault(key, []).append(row)

    for (pair, feed, e_core), members in groups.items():
        pats = [unpack_pattern(str(r["pattern"])) for r in members]
        pred = backend.predict(pats, CaseKey(pair, feed), float(e_core))
        cols = {name: np.asarray(pred.mean[:, COL[name]], dtype=float)
                for name in names}
        sigma = np.asarray(pred.calibrated_std[:, COL["max_pin_burnup"]], dtype=float)
        try:                                    # quantile heads are optional
            lo, hi = pred.band("max_pin_burnup")
            band = (np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
        except Exception:                                   # noqa: BLE001
            band = (np.full(len(members), np.nan), np.full(len(members), np.nan))
        for i, row in enumerate(members):
            out.append({
                "record_id": str(row["record_id"]),
                "campaign": str(row["campaign"]),
                "case_pair": pair,
                "feed": feed,
                "library_id": library_id,
                "e_core": float(e_core),
                "e_split": (None if not _finite(row.get("e_split"))
                            else float(row["e_split"])),
                "pattern": str(row["pattern"]),
                "restart_provenance": str(row.get("restart_provenance") or ""),
                "stored": {k: (None if not _finite(row.get(k)) else float(row[k]))
                           for k in ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                                     "node_peak", "map_cov",
                                     "max_assembly_burnup", "max_pin_burnup",
                                     "n_cycles")},
                "predicted": {k: float(v[i]) for k, v in cols.items()},
                "predicted_pin_sigma": float(sigma[i]),
                "predicted_pin_band": [float(band[0][i]), float(band[1][i])],
            })
    return out


def _select_delivery(df, group: dict) -> list:
    sub = df[df.campaign == group["campaign"]]
    if sub.empty:
        raise SystemExit(f"no store rows for campaign {group['campaign']!r}")
    keep = [r for _, r in sub.iterrows() if joint_clean(r, group["f_r_limit"])]
    if not keep:
        raise SystemExit(f"{group['name']}: no joint-clean row "
                         f"(cbc<={CBC_LIMIT}, f_q<={F_Q_LIMIT}, "
                         f"|ao|<={AO_ABS_LIMIT}, f_r<={group['f_r_limit']})")
    keep.sort(key=lambda r: _f(r["f_r"]))
    import pandas as pd
    return pd.DataFrame(keep[: int(group["top_n"])])


def _select_calibration(df, cell: dict, backend, exclude: set[str]) -> list:
    """Stratified span over the PREDICTED pin axis for one pin-pessimism cell.

    A calibration CURVE needs the predicted axis covered, so this cuts the cell's
    converged, not-already-measured rows into ``bins`` equal-width predicted-pin
    bins over the cell's own ``[pin_lo, pin_hi]`` window and draws ``per_bin``
    core(s) from each.

    Within a bin the LOWEST stored F_r wins (ties -> nearest the bin centre).
    Binning already fixes the spread on the axis being calibrated, so the
    within-bin choice is free, and it is spent deliberately: the residual this
    wave has to estimate is the one that applies WHERE THE VERDICTS LIVE.  The
    store's low-feed cells are dominated by unoptimized produce cores (F_r up to
    4.1) that no delivery decision will ever reference, and the standing
    hypothesis from ``pinbu_forensics.md`` is precisely that the head's error
    grows off the training manifold -- so a residual measured only on those cores
    would be the wrong number.  This is a deliberate, declared selection on F_r;
    it is legitimate only because the regression is measured-pin on
    PREDICTED-pin, and the bins (not the F_r ranking) set that regressor's span.

    ``exclude`` holds record_ids already planned (typically this cell's own
    delivery winners) so one core is never chained twice.
    """
    import numpy as np
    import pandas as pd

    sub = df[(df.case_pair == cell["case_pair"]) & (df.feed == int(cell["feed"]))]
    if cell.get("campaign"):
        sub = sub[sub.campaign == cell["campaign"]]
    sub = sub[sub.valid.fillna(False) & sub.converged.fillna(False)
              & sub.f_r.notna() & sub.max_pin_burnup.isna()
              & ~sub.record_id.astype(str).isin(exclude)]
    if sub.empty:
        raise SystemExit(f"calibration cell {cell['name']} has no eligible store row")

    scored = _score(backend, sub, cell["library_id"])
    by_rid = {s["record_id"]: s["predicted"]["max_pin_burnup"] for s in scored}
    sub = sub.assign(_pred_pin=[by_rid[str(r)] for r in sub["record_id"]])

    lo, hi = float(cell["pin_lo"]), float(cell["pin_hi"])
    bins, per_bin = int(cell["bins"]), int(cell.get("per_bin", 1))
    edges = np.linspace(lo, hi, bins + 1)
    picked: list = []
    taken: set[str] = set()
    for b in range(bins):
        centre = 0.5 * (edges[b] + edges[b + 1])
        mask = (sub["_pred_pin"] >= edges[b]) & (
            sub["_pred_pin"] <= edges[b + 1] if b == bins - 1
            else sub["_pred_pin"] < edges[b + 1])
        cand = sub[mask & ~sub.record_id.astype(str).isin(taken)]
        if cand.empty:
            continue
        order = cand.assign(_d=(cand["_pred_pin"] - centre).abs()) \
                    .sort_values(["f_r", "_d"], kind="stable").index
        chosen = cand.loc[order[:per_bin]]
        taken.update(chosen["record_id"].astype(str))
        picked.extend(chosen.drop(columns=["_pred_pin"]).to_dict("records"))
    if not picked:
        raise SystemExit(f"calibration cell {cell['name']}: no store row landed in "
                         f"predicted pin [{lo},{hi}]")
    return pd.DataFrame(picked)


def cmd_plan(args) -> int:
    import pandas as pd
    import torch
    torch.set_num_threads(max(1, int(args.threads)))
    from lpopt.model.model_api import PosValCnnBackend

    store_dir = Path(args.store_dir)
    store_dir = store_dir if store_dir.is_absolute() else BASE / store_dir
    df = load_store(store_dir)

    calib_cells = list(CALIB_CELLS)
    if args.calib_spec:
        calib_cells = json.loads(Path(args.calib_spec).read_text(encoding="utf-8"))

    backends: dict[str, object] = {}

    def backend_for(library_id: str):
        if library_id not in backends:
            print(f"[plan] loading {args.model} for library {library_id} ...",
                  flush=True)
            backends[library_id] = PosValCnnBackend.from_dir(
                Path(args.model) if Path(args.model).is_absolute()
                else BASE / args.model,
                store_dir=store_dir, library_id=library_id, device="cpu")
        return backends[library_id]

    targets: list[dict] = []
    groups_meta: list[dict] = []

    for g in DELIVERY_GROUPS:
        sel = _select_delivery(df, g)
        scored = _score(backend_for(g["library_id"]), sel, g["library_id"])
        for rank, s in enumerate(sorted(scored, key=lambda s: s["stored"]["f_r"]), 1):
            s["group"] = g["name"]
            s["role"] = "delivery"
            s["rank_in_group"] = rank
            targets.append(s)
        groups_meta.append({**g, "kind": "delivery", "n_selected": len(scored)})

    # Delivery cores are planned first, so a calibration cell that overlaps a
    # delivery cell (N1_N2/f113, E1_E2/f109) draws AROUND them instead of
    # re-chaining a core the wave is already measuring.
    planned_ids = {t["record_id"] for t in targets}
    for cell in calib_cells:
        sel = _select_calibration(df, cell, backend_for(cell["library_id"]),
                                  planned_ids)
        scored = _score(backend_for(cell["library_id"]), sel, cell["library_id"])
        for s in sorted(scored, key=lambda s: s["predicted"]["max_pin_burnup"]):
            s["group"] = cell["name"]
            s["role"] = "calibration"
            s["rank_in_group"] = 0
            targets.append(s)
            planned_ids.add(s["record_id"])
        groups_meta.append({**cell, "kind": "calibration", "n_selected": len(scored)})

    plan = {
        "schema": PLAN_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(args.model),
        "store_records_bytes": int((store_dir / "records.parquet").stat().st_size),
        "licensing_pin_limit": LICENSING_PIN_LIMIT,
        "acquisition_pin_limit": ACQUISITION_PIN_LIMIT,
        "determinism_tolerances": {"f_r": TOL_F_R, "cyclen": TOL_CYCLEN,
                                   "cbc_max": TOL_CBC},
        "joint_gates": {"cbc_max": CBC_LIMIT, "f_q": F_Q_LIMIT,
                        "ao_abs": AO_ABS_LIMIT},
        "groups": groups_meta,
        "n_targets": len(targets),
        "targets": targets,
    }
    out = Path(args.out) if Path(args.out).is_absolute() else BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"\n[plan] {len(targets)} target core(s) -> {out}")
    for g in groups_meta:
        rows = [t for t in targets if t["group"] == g["name"]]
        if not rows:
            continue
        pins = [t["predicted"]["max_pin_burnup"] for t in rows]
        print(f"  {g['name']:22s} {g['kind']:11s} n={len(rows):2d}  "
              f"predicted pin {min(pins):.2f}..{max(pins):.2f}")
    return 0


# --------------------------------------------------------------------------- #
# RUN — one pin-burnup chain per planned core
# --------------------------------------------------------------------------- #
def _verifier_for(cfg, library_id: str, run_dir: Path, fuel_library):
    from lpopt.curriculum import make_pin_burnup_verifier
    from lpopt.search.resolver import build_case_resolver

    resolver = build_case_resolver(cfg, fuel_library, library_id)
    verifier = make_pin_burnup_verifier(cfg, run_dir / library_id, resolver)
    return resolver, verifier


def cmd_run(args) -> int:
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.data.schema import compute_record_id, unpack_pattern
    from lpopt.search.verify import PRODUCE_DECK_KNOBS, WaveEntry
    from lpopt.vendor.masterrl.domain import CaseKey

    plan_path = Path(args.plan) if Path(args.plan).is_absolute() else BASE / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"{plan_path} is not a {PLAN_SCHEMA} plan")
    targets = plan["targets"]
    if args.group:
        targets = [t for t in targets if t["group"] in set(args.group)]
    if args.record_id:
        want = tuple(args.record_id)
        targets = [t for t in targets if t["record_id"].startswith(want)]
    if not targets:
        raise SystemExit("no planned target matched the filters")

    deck = Path(args.deck) if Path(args.deck).is_absolute() else BASE / args.deck
    cfg = load_config(deck)
    run_dir = Path(args.run_dir) if Path(args.run_dir).is_absolute() else BASE / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    store_dir = Path(cfg.model.store_dir)
    store_dir = store_dir if store_dir.is_absolute() else (deck.resolve().parent / store_dir)
    fuel_library = FuelLibrary.from_parquet(store_dir / "fuel_types.parquet")

    results_path = run_dir / RESULTS_NAME
    # RESUME: a completed chain is never re-spent.  The results file is the
    # ledger (the run dir's MASTER work is purged), so it is the only thing that
    # can answer "did this core already run".
    done: set[str] = set()
    if results_path.is_file() and not args.force:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["record_id"])
            except (ValueError, KeyError):
                continue
    todo = [t for t in targets if t["record_id"] not in done]
    print(f"deck     : {deck}")
    print(f"exe      : {cfg.master.executable}")
    print(f"run dir  : {run_dir}")
    print(f"planned  : {len(targets)}  already done: {len(targets)-len(todo)}  "
          f"to run: {len(todo)}")
    if not todo:
        print("nothing to run")
        return 0

    by_library: dict[str, list[dict]] = {}
    for t in todo:
        by_library.setdefault(t["library_id"], []).append(t)

    t_all = time.monotonic()
    n_ok = n_det = 0
    for library_id, group in by_library.items():
        resolver, verifier = _verifier_for(cfg, library_id, run_dir, fuel_library)
        entries: list[WaveEntry] = []
        skipped: list[dict] = []
        for t in group:
            pattern = unpack_pattern(t["pattern"])
            key = CaseKey(t["case_pair"], int(t["feed"]))
            # IDENTITY GATE, before any MASTER call.  ``outcome_to_record`` mints
            # the record_id from (canonical pattern, library_id, pair, deck_knobs);
            # if that does not reproduce the planned id, this chain would be
            # written as a NEW store row instead of measuring the stored core --
            # the exact failure ``ablation_wave.py:848`` guards against, caught
            # here where it costs nothing.
            minted = compute_record_id(pattern.canonical(), library_id,
                                       t["case_pair"], PRODUCE_DECK_KNOBS)
            if minted != t["record_id"]:
                skipped.append({**t, "failure": f"record_id drift: plan "
                                f"{t['record_id'][:12]} vs minted {minted[:12]}"})
                continue
            assets = resolver.resolve(key)
            entries.append(WaveEntry(pattern, key, assets, {
                "record_id": t["record_id"], "group": t["group"],
                "role": t["role"], "e_core": t["e_core"],
                "planned_provenance": t["restart_provenance"],
            }))
        for s in skipped:
            print(f"  [SKIP] {s['record_id'][:12]} {s['failure']}")
        if not entries:
            continue

        prov = sorted({e.resolved_assets.restart_provenance for e in entries})
        print(f"\n=== library {library_id}: {len(entries)} chain(s), "
              f"workers={verifier.n_workers}, restart provenance {prov}")
        if args.dry_run:
            for e in entries:
                print(f"    [dry-run] {e.meta['record_id'][:12]} {e.case_key.label} "
                      f"fallback={e.resolved_assets.fallback_level} "
                      f"restart={e.resolved_assets.restart_provenance}")
            continue

        t0 = time.monotonic()
        outcomes = verifier.evaluate_wave(entries)
        print(f"    wave wall {time.monotonic()-t0:.0f}s")

        planned = {t["record_id"]: t for t in group}
        with results_path.open("a", encoding="utf-8") as fh:
            for oc in outcomes:
                rec = _outcome_record(oc, planned[oc.meta["record_id"]])
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_ok += int(rec["status"] == "converged")
                n_det += int(bool(rec["determinism_ok"]))
                pin = rec["measured"]["max_pin_burnup"]
                pin_s = "  --  " if pin is None else f"{pin:6.3f}"
                print(f"    {rec['record_id'][:12]} {rec['group']:20s} "
                      f"{rec['status']:12s} pin={pin_s}  "
                      f"det_ok={rec['determinism_ok']}  {rec['wall_s']:.0f}s")

    print(f"\n{'='*78}\n{n_ok} converged / {n_det} determinism-ok, "
          f"total wall {time.monotonic()-t_all:.0f}s -> {results_path}")
    return 0


def _outcome_record(outcome, planned: dict) -> dict:
    """One results-JSONL row: the pinned prediction, the measurement, the control."""
    fom = outcome.fom
    measured = {
        name: (None if fom is None or not _finite(getattr(fom, name, None))
               else float(getattr(fom, name)))
        for name in ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                     "max_assembly_burnup", "max_pin_burnup")
    }
    stored = planned["stored"]
    deltas: dict[str, float | None] = {}
    for name in ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
                 "max_assembly_burnup"):
        s, m = stored.get(name), measured.get(name)
        deltas[name] = None if (s is None or m is None) else float(m - s)
    det = (outcome.status == "converged"
           and deltas["f_r"] is not None and abs(deltas["f_r"]) <= TOL_F_R
           and deltas["cyclen"] is not None and abs(deltas["cyclen"]) <= TOL_CYCLEN
           and deltas["cbc_max"] is not None and abs(deltas["cbc_max"]) <= TOL_CBC)
    pin = measured["max_pin_burnup"]
    return {
        "record_id": planned["record_id"],
        "group": planned["group"],
        "role": planned["role"],
        "campaign": planned["campaign"],
        "case_pair": planned["case_pair"],
        "feed": planned["feed"],
        "library_id": planned["library_id"],
        "e_core": planned["e_core"],
        "status": outcome.status,
        "n_cycles": int(outcome.n_cycles),
        "converged_at_cap": bool(outcome.converged_at_cap),
        "tolerance_margin": outcome.tolerance_margin,
        "restart_provenance": outcome.restart_provenance,
        "planned_provenance": planned["restart_provenance"],
        "provenance_ok": (outcome.restart_provenance
                          == planned["restart_provenance"]),
        "failure": outcome.failure,
        "wall_s": round(float(outcome.wall_s), 1),
        "core_class": outcome.core_class,
        "stored": stored,
        "predicted": planned["predicted"],
        "predicted_pin_band": planned.get("predicted_pin_band"),
        "measured": measured,
        "deltas_measured_minus_stored": deltas,
        "determinism_ok": bool(det),
        # The two readouts this wave exists for.  Both are None when the chain
        # did not produce a pin value -- never a guess.
        "pin_error_pred_minus_meas": (
            None if pin is None
            or not _finite(planned["predicted"].get("max_pin_burnup"))
            else float(planned["predicted"]["max_pin_burnup"] - pin)),
        "delivery_verdict": (
            None if pin is None else
            ("PASS" if pin <= LICENSING_PIN_LIMIT else "FAIL")),
    }


# --------------------------------------------------------------------------- #
# PATCH — merge the measured column into the canonical store, IN PLACE
# --------------------------------------------------------------------------- #
def cmd_patch(args) -> int:
    """Write measured ``max_pin_burnup`` onto the EXISTING store rows.

    Why in place and not ``write_records`` / ``lpopt merge-store``:
    ``store.dedup_upsert`` ranks a duplicate record_id on
    ``converged*4 + valid*2 + has_flatness`` -- a rank that does NOT include
    ``max_pin_burnup``.  A re-evaluated row that differs ONLY by carrying the pin
    value ranks EQUAL to the stored one, so ``multi_pc.merge_store`` classifies it
    ``duplicate``, leaves ``changed`` False and never writes
    (``multi_pc.py:1374``); and a direct ``write_records`` would replace the whole
    row -- discarding the stored ``maps_key`` / ``node_peak`` / ``map_cov`` if this
    wave ran without map harvest.  ``lpopt/tools/backfill_flatness.py`` is the
    precedent for populating a nullable column on rows that already exist: patch
    by record_id, preserve row order, atomic write.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from lpopt.data.store import (
        RECORDS_NAME, _atomic_write, ensure_schema_columns, frame_to_table)

    results = Path(args.results) if Path(args.results).is_absolute() else BASE / args.results
    store_dir = Path(args.store_dir) if Path(args.store_dir).is_absolute() else BASE / args.store_dir
    rows = [json.loads(ln) for ln in results.read_text(encoding="utf-8").splitlines() if ln.strip()]

    accept: dict[str, float] = {}
    accept_asm: dict[str, float] = {}
    refused: list[tuple[str, str]] = []
    for r in rows:
        pin = r["measured"].get("max_pin_burnup")
        if r["status"] != "converged":
            refused.append((r["record_id"], f"status={r['status']}"))
            continue
        if pin is None:
            refused.append((r["record_id"], "no PPI pin value in the chain"))
            continue
        if not r["determinism_ok"]:
            # The replay did not reproduce the stored labels, so the pin value
            # measured a DIFFERENT evaluation than the one the store row records.
            # Attaching it to that row would silently mix two chains.
            refused.append((r["record_id"], "determinism_ok=False "
                            f"(d {r['deltas_measured_minus_stored']})"))
            continue
        if not r["provenance_ok"]:
            refused.append((r["record_id"], "restart provenance changed "
                            f"({r['planned_provenance']} -> {r['restart_provenance']})"))
            continue
        accept[r["record_id"]] = float(pin)
        asm = r["measured"].get("max_assembly_burnup")
        if asm is not None:
            accept_asm[r["record_id"]] = float(asm)

    print(f"[patch] {len(rows)} result row(s): {len(accept)} accepted, "
          f"{len(refused)} refused")
    for rid, why in refused:
        print(f"  [refused] {rid[:12]}  {why}")
    if not accept:
        print("[patch] nothing to write")
        return 0

    path = store_dir / RECORDS_NAME
    current = ensure_schema_columns(pd.read_parquet(path))
    rid_col = current["record_id"].astype(str)
    present = set(rid_col) & set(accept)
    missing = sorted(set(accept) - present)
    if missing:
        # A measured pin with no store row to attach it to is a planning error
        # (wrong deck knobs / library), never something to paper over by
        # appending a new row: that would create a SECOND row for one core.
        raise SystemExit(f"[patch] {len(missing)} measured core(s) are not in the "
                         f"store: {[m[:12] for m in missing]}")

    overwrites = [rid for rid in accept
                  if pd.notna(current.loc[rid_col == rid, "max_pin_burnup"]).any()]
    if overwrites and not args.allow_overwrite:
        raise SystemExit(f"[patch] {len(overwrites)} row(s) already carry a "
                         f"max_pin_burnup: {[o[:12] for o in overwrites]} -- pass "
                         f"--allow-overwrite to replace measured values")

    print(f"[patch] would write max_pin_burnup on {len(accept)} row(s) "
          f"({len(accept_asm)} also carry max_assembly_burnup)")
    if args.dry_run:
        print("[patch] dry run; store untouched")
        return 0

    backup = path.with_name(f"{path.name}.bak_pre_{args.tag}")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
        print(f"[patch] backup -> {backup.name}")

    for column, values in (("max_pin_burnup", accept),
                           ("max_assembly_burnup", accept_asm)):
        patched = rid_col.map(values)
        current[column] = patched.where(
            patched.notna(), pd.to_numeric(current[column], errors="coerce"))
    table = frame_to_table(current)
    _atomic_write(path, lambda p: pq.write_table(table, p))
    print(f"[patch] wrote {len(accept)} measured pin value(s); "
          f"store now {len(current)} row(s)")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pinbu_wave.py",
        description="measured pin-burnup re-evaluation of stored cores")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="select targets + pin the prediction (read-only)")
    p.add_argument("--model", default="data/models/s1i")
    p.add_argument("--store-dir", default="data/store")
    p.add_argument("--calib-spec", default=None,
                   help="JSON list of calibration cells (see CALIB_CELLS)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="data/reports/pinbu_wave_prereg_20260820.json")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run one pin-burnup chain per planned core")
    p.add_argument("--plan", default="data/reports/pinbu_wave_prereg_20260820.json")
    p.add_argument("--deck", default="pinbu_wave_199.inp")
    p.add_argument("--run-dir", default="runs/pinbu_wave")
    p.add_argument("--group", action="append", default=None)
    p.add_argument("--record-id", action="append", default=None)
    p.add_argument("--force", action="store_true",
                   help="re-run cores that already have a results row")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve assets and print the plan; start no MASTER")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("patch", help="write measured pin burnup into the store")
    p.add_argument("--results", default="runs/pinbu_wave/" + RESULTS_NAME)
    p.add_argument("--store-dir", default="data/store")
    p.add_argument("--tag", default="pinbu_20260820")
    p.add_argument("--allow-overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_patch)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
