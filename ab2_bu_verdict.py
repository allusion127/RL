"""AB2 round 4 — arm BU (burnup placement): score the pre-registered gate.

Executes the rule fixed in `data/reports/ab2_addendum_BU_20260810.md` §5, which
was written and saved before either arm was trained.  Nothing here chooses a
threshold: every number below is transcribed from that document.

Rebuilt from the round-1..3 apparatus (`ab2_verdict.py` / `ab2_e10.py` /
`ab2_r3.py`, since deleted from the scratchpad) — same estimator, same surface,
same control discipline: served CSVs from the GPU box, statistics on this box.

PREREQUISITE — produce the two served CSVs on the training box first:

    ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
        ./venv/bin/python eval_accuracy.py runs/bu_A0 bu_A0 && \
        ./venv/bin/python eval_accuracy.py runs/bu_T  bu_T'

    cd "<5_RL>/runs/ab2_bu"
    scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_bu_A0.csv .
    scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_bu_T.csv  .

THEN, from anywhere:

    python "<5_RL>/ab2_bu_verdict.py"

Writes `data/reports/ab2_verdict_BU_20260810.json` and prints the verdict.
Env overrides for testing: AB2_BU_DIR (input dir), AB2_BU_OUT (output json).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# paths — the two constants this script has
# --------------------------------------------------------------------------- #
#: This file lives at the repo root (5_RL/), so the repo is simply its parent.
#: Deliberately NOT a hard-coded absolute path: rounds 1-3 kept their scorers in
#: a session scratchpad with the repo path baked in, and every one of them was
#: wiped in a cleanup and had to be recovered from a transcript.
REPO = Path(__file__).resolve().parent

#: Where the two served CSVs land.  Overridable so the pipeline can be smoke-
#: tested on synthetic arms without colliding with the real filenames.
SCR = Path(os.environ.get("AB2_BU_DIR") or REPO / "runs" / "ab2_bu")

sys.path.insert(0, str(REPO))
os.chdir(REPO)

from lpopt.model import flat_ab as FA                    # noqa: E402
from lpopt.model import flat_metrics as FM               # noqa: E402
from lpopt.model.ab_paired import MDE_POWER              # noqa: E402

CONTROL, TREATED = "A0BU", "TBU"
#: Filenames are overridable so later rounds can reuse this apparatus unchanged
#: (round 6 / ADF scores `rows_split_champ.csv` vs `rows_adf_cand.csv`).
CSV = {
    CONTROL: SCR / (os.environ.get("AB2_CONTROL_CSV") or "rows_bu_A0.csv"),
    TREATED: SCR / (os.environ.get("AB2_TREATED_CSV") or "rows_bu_T.csv"),
}

# --- the rule, transcribed from the addendum ------------------------------- #
REPS, SEED, ALPHA = 2000, 0, 0.05                        # §5.1
PRIMARY = "T_cell_mae_node_peak"                         # §5.2 — ONE axis
MDE80_BAR = 0.00409                                      # §5.3, from round 1
HARM = {                                                 # §5.4
    "T_cell_mae_cyclen": 0.10,        # EFPD
    "T_cell_mae_cbc_max": 1.0,        # ppm
    "M7_cell_mae_map_cov": 0.002,
}
VARIANCE_AXES = ("T_cell_mae_node_peak", "M7_cell_mae_map_cov",
                 "T_cell_mae_f_q", "T_cell_mae_cbc_max", "T_cell_mae_cyclen")
SEED_REPRO = (0, 1, 2, 3, 4)                             # §5.7

# --- frozen snapshot pins, from addendum §3 -------------------------------- #
PIN = {
    "records.parquet":
        "4039bc96cffb52fb3f8c371aa94127fb17e70765775b452c6b29ae6068eeed3f",
    "S1.json":
        "6ab35f25027c8355fe4f07bda2a200b1c5baf40d445257e255ed89dead650640",
}
TARGETS = ["cyclen", "cbc_max", "f_q", "node_peak", "map_cov", "f_r"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _spec(key: str):
    return FM.AB2_METRICS_BY_KEY.get(key) or FM.METRICS_BY_KEY[key]


def main() -> int:
    OUT: dict = {"schema": "ab2_BU_verdict_v1",
                 "rule_doc": "data/reports/ab2_addendum_BU_20260810.md",
                 "checks": {}}
    C = OUT["checks"]

    missing = [str(p) for p in CSV.values() if not p.is_file()]
    if missing:
        print("MISSING served CSVs — run eval_accuracy.py on the box first:")
        for m in missing:
            print("   ", m)
        print(__doc__.split("PREREQUISITE")[1].split("THEN")[0])
        return 2

    # -- 0. snapshot pins --------------------------------------------------- #
    C["records_sha256_now"] = _sha(REPO / "data/store/records.parquet")
    C["records_sha256_pinned"] = PIN["records.parquet"]
    C["records_matches_pin"] = C["records_sha256_now"] == PIN["records.parquet"]
    C["S1_sha256_now"] = _sha(REPO / "data/splits/S1.json")
    C["S1_matches_pin"] = C["S1_sha256_now"] == PIN["S1.json"]

    # -- 1. load + comparability (§5.8) ------------------------------------- #
    raw = {a: pd.read_csv(p) for a, p in CSV.items()}
    C["row_counts"] = {a: int(len(d)) for a, d in raw.items()}
    C["columns_identical"] = (list(raw[CONTROL].columns)
                              == list(raw[TREATED].columns))
    ids = {a: d["record_id"].astype(str) for a, d in raw.items()}
    C["same_id_set"] = set(ids[CONTROL]) == set(ids[TREATED])
    C["n_control_only"] = len(set(ids[CONTROL]) - set(ids[TREATED]))
    C["n_treated_only"] = len(set(ids[TREATED]) - set(ids[CONTROL]))
    C["dup_ids"] = {a: int(s.duplicated().sum()) for a, s in ids.items()}
    C["same_id_order_as_delivered"] = bool(
        (ids[CONTROL].to_numpy() == ids[TREATED].to_numpy()).all()
        if C["row_counts"][CONTROL] == C["row_counts"][TREATED] else False)

    if not C["same_id_set"] or any(C["dup_ids"].values()):
        OUT["verdict"] = {"verdict": "ESCALATE",
                          "reason": "row sets not comparable (§5.8 checks 1-3)"}
        _write(OUT)
        return 1

    # THE ALIGNMENT IS THE PAIRING — reindex TREATED onto CONTROL's order.
    base = raw[CONTROL].reset_index(drop=True)
    other = (raw[TREATED].set_index(raw[TREATED]["record_id"].astype(str))
             .loc[base["record_id"].astype(str)].reset_index(drop=True))
    raw[TREATED] = other
    C["reindexed_treated_onto_control_order"] = True
    C["order_after_reindex"] = bool(
        (base["record_id"].astype(str).to_numpy()
         == other["record_id"].astype(str).to_numpy()).all())

    truth_cols = [c for c in base.columns if c.startswith("true_")]
    C["truth_identical_across_arms"] = all(
        np.allclose(pd.to_numeric(base[c], errors="coerce"),
                    pd.to_numeric(other[c], errors="coerce"),
                    equal_nan=True) for c in truth_cols)

    # -- 2. the frozen 36-cell surface, read from the split ------------------ #
    split_name = os.environ.get("AB2_SPLIT") or "S1"
    split = json.loads(
        (REPO / f"data/splits/{split_name}.json").read_text(encoding="utf-8"))
    # THE FROZEN SURFACE, NOT THE GROWN ONE.  The S1b rebuild added 120 val rows
    # to three of the 36 cells, so `curriculum_val_by_cell` is no longer the
    # 3,207-row surface rounds 1-3 and BU were judged on -- and MDE80 = 0.00409
    # is only comparable on that surface.  `ab2_frozen_val_by_cell` preserves it
    # verbatim (ab2_addendum_SPLIT_20260810.md §3.3); prefer it whenever present.
    groups = split["groups"]
    cv = groups.get("ab2_frozen_val_by_cell") or groups["curriculum_val_by_cell"]
    C["surface_key"] = ("ab2_frozen_val_by_cell"
                        if "ab2_frozen_val_by_cell" in groups
                        else "curriculum_val_by_cell")
    C["surface_split"] = split_name
    frozen_cells = tuple(sorted(cv))
    cell_of = {rid: c for c, rids in cv.items() for rid in rids}
    C["frozen_n_cells"] = len(frozen_cells)
    C["frozen_n_rows"] = sum(len(v) for v in cv.values())
    rid = base["record_id"].astype(str)
    in_frozen = rid.isin(cell_of)
    C["frozen_rows_served"] = int(in_frozen.sum())
    C["cell_csv_equals_split_cell_on_frozen"] = bool(
        (base.loc[in_frozen, "cell"].astype(str).to_numpy()
         == rid[in_frozen].map(cell_of).to_numpy()).all())

    # -- 3. arena ------------------------------------------------------------ #
    cells = base["cell"].astype(str).to_numpy()
    truth = {t: pd.to_numeric(base.get(f"true_{t}"), errors="coerce").to_numpy(float)
             for t in TARGETS if f"true_{t}" in base.columns}
    preds = {a: {t: pd.to_numeric(d.get(f"pred_{t}"), errors="coerce").to_numpy(float)
                 for t in truth} for a, d in raw.items()}
    arena = FA.FlatArena(
        cells=cells, truth=truth, preds=preds, control=CONTROL,
        record_ids=tuple(rid), frozen_cells=frozen_cells,
        provenance={"split": "S1", "surface": "curriculum_val_by_cell",
                    "n_cells": len(frozen_cells), "rule": OUT["rule_doc"]})

    def paired(key: str, seed: int = SEED):
        return FA.paired_metric(arena, TREATED, _spec(key), reps=REPS,
                                seed=seed, alpha=ALPHA)

    # -- 4. condition 1 — the gate (§5.3) ------------------------------------ #
    d = paired(PRIMARY)
    mde = d.mde()
    gate = {
        "metric": PRIMARY, "point": d.point, "ci_lo": d.ci_lo, "ci_hi": d.ci_hi,
        "se": d.se if hasattr(d, "se") else None, "method": d.method,
        "n_cells": d.n_cells, "mde80_measured": None if not np.isfinite(mde) else mde,
        "mde80_bar_preregistered": MDE80_BAR, "mde_power": MDE_POWER,
        "clause_point_ge_mde80": bool(d.point >= MDE80_BAR),
        "clause_ci_excludes_zero": bool(d.establishes_gain()),
    }
    gate["PASS"] = bool(gate["clause_point_ge_mde80"]
                        and gate["clause_ci_excludes_zero"])
    gate["underpowered"] = bool(abs(d.point) < (mde if np.isfinite(mde) else np.inf))
    OUT["condition1_gate"] = gate

    # -- 5. condition 2 — harm bounds (§5.4) --------------------------------- #
    harms: dict = {}
    for key, eps in HARM.items():
        h = paired(key)
        harms[key] = {"epsilon": eps, "point": h.point, "ci_lo": h.ci_lo,
                      "ci_hi": h.ci_hi, "harm_upper": h.harm_upper,
                      "status": FA._harm_status(h, eps),
                      "bounded": bool(h.bounds_harm(eps))}
    for key, inherited in FA.HARM_MARGINS.items():
        h = paired(key)
        # `M7_cell_mae_map_cov` appears in BOTH sets: the addendum §5.4 pins it
        # at 0.002 and the inherited flat_ab rail at 0.005.  The STRICTER one
        # binds -- letting the inherited rail overwrite the arm-specific bound
        # would silently relax the pre-registered rule by 2.5x.
        eps = min(inherited, harms[key]["epsilon"]) if key in harms else inherited
        harms[key] = {"epsilon": eps, "epsilon_addendum": HARM.get(key),
                      "epsilon_inherited_rail": inherited,
                      "point": h.point, "ci_lo": h.ci_lo,
                      "ci_hi": h.ci_hi, "harm_upper": h.harm_upper,
                      "status": FA._harm_status(h, eps),
                      "enforced": key != FA.FR_HARM_METRIC,
                      "bounded": bool(h.bounds_harm(eps))}
    OUT["condition2_harm"] = harms

    established_worse = []
    for key in VARIANCE_AXES:
        v = paired(key)
        if np.isfinite(v.ci_hi) and v.ci_hi < 0.0:
            established_worse.append(key)
    OUT["established_worse_axes"] = established_worse

    violated = [k for k, v in harms.items()
                if v.get("enforced", True) and v["status"] == "violated"]

    # -- 6. every axis, reported ------------------------------------------- #
    OUT["all_axes"] = {}
    for key in list(FM.AB2_METRICS_BY_KEY) + list(FA.HARM_MARGINS):
        try:
            a = paired(key)
        except (KeyError, ValueError):
            continue
        OUT["all_axes"][key] = a.to_dict()

    # -- 7. verdict (§5.5) --------------------------------------------------- #
    if violated or established_worse:
        verdict, why = "REJECT", (
            f"harm violated {violated}; established worse {established_worse}")
    elif gate["PASS"]:
        verdict, why = "PASS", "§5.3 both clauses hold and no harm bound exceeded"
    else:
        verdict, why = "HOLD", (
            "§5.3 unmet (point %.5f vs bar %.5f; ci_lo %.5f)"
            % (gate["point"], MDE80_BAR, gate["ci_lo"]))
    if d.method in ("insufficient", "degenerate"):
        verdict, why = "ESCALATE", f"primary axis returned method={d.method}"
    OUT["verdict"] = {"verdict": verdict, "reason": why}

    # -- 8. seed reproduction (§5.7) ----------------------------------------- #
    repro_keys = {PRIMARY} | set(violated) | set(established_worse)
    OUT["seed_reproduction"] = {
        k: {str(s): paired(k, seed=s).to_dict() for s in SEED_REPRO}
        for k in sorted(repro_keys)}

    # -- 9. falsification, transcribed (§5.6) -------------------------------- #
    OUT["falsification"] = {
        "fired": verdict != "PASS",
        "consequence": (
            "memo §3(0) 'fix the burnup placement error first' is REJECTED; only "
            "the curve-precision half of v7 remains, and because KILLER 1 still "
            "stands the whole of v7 is re-examined rather than continued."),
        "note": ("§3.4: the mechanism acts on every scored row in both libraries "
                 "of the surface, so a null here is NOT excusable as a "
                 "surface/mechanism mismatch the way R3's was."),
    }

    _write(OUT)
    _print(OUT)
    return 0


def _write(OUT: dict) -> None:
    p = Path(os.environ.get("AB2_BU_OUT")
             or REPO / "data/reports/ab2_verdict_BU_20260810.json")
    p.write_text(json.dumps(OUT, indent=1, ensure_ascii=False, default=float),
                 encoding="utf-8")
    print(f"\n  -> {p}")


def _print(OUT: dict) -> None:
    g = OUT["condition1_gate"]
    print("\n=== AB2 round 4 — arm BU ===")
    print(f"control {CONTROL}  treated {TREATED}  "
          f"{OUT['checks']['frozen_n_cells']} frozen cells / "
          f"{OUT['checks']['frozen_rows_served']} rows")
    print(f"\nGATE  {g['metric']}")
    print(f"  point {g['point']:+.5f}   CI [{g['ci_lo']:+.5f}, {g['ci_hi']:+.5f}]"
          f"   method {g['method']}")
    print(f"  MDE80 measured {g['mde80_measured']}   bar {g['mde80_bar_preregistered']}")
    print(f"  clause point>=MDE80 : {g['clause_point_ge_mde80']}")
    print(f"  clause ci_lo>0      : {g['clause_ci_excludes_zero']}")
    print("\nHARM")
    for k, v in OUT["condition2_harm"].items():
        if v.get("enforced", True):
            print(f"  {k:38s} eps {v['epsilon']:<7} upper "
                  f"{v['harm_upper']!s:>10}  {v['status']}")
    print(f"\nestablished worse: {OUT['established_worse_axes'] or 'none'}")
    print(f"\nVERDICT: {OUT['verdict']['verdict']} — {OUT['verdict']['reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
