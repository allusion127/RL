"""Round 5 arm SPLIT — the §5.3 registered SECONDARY readout.

Pre-registered in `data/reports/ab2_addendum_SPLIT_20260810.md` §5.3:

    within-cell served `f_r` Spearman on the `f_r <= 1.55` slice,
    champion vs candidate, on the same val rows.

**This decides nothing.** The primary instrument is `lpopt gate-promote`
(§5.1/§5.2). This number is reported either way because the ranking blindness is
the question that motivated the arm, and §5.3 fixed in advance both how it is
measured and how weak it is expected to be.

PREREQUISITE — serve BOTH models on the SAME split (S1b) on the GPU box:

    ssh -p 8022 USER@HOST_238 'cd $HOME/lpopt_ws && \
        ./venv/bin/python eval_accuracy_split.py runs/split_S1b split_cand  S1b && \
        ./venv/bin/python eval_accuracy_split.py runs/bu_T      split_champ S1b'

(`runs/bu_T` on the box IS `data/models/20260810_bu_T` locally — the champion was
trained there. Both models are served on S1b, so the comparison is the same rows
in the same order; neither trained on the 120 new val rows.)

    cd 5_RL/runs/split_S1b
    scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_split_cand.csv .
    scp -P 8022 USER@HOST_238:lpopt_ws/reports/rows_split_champ.csv .

THEN:  python 5_RL/split_secondary_readout.py

Env overrides for testing: SPLIT_RO_DIR (input dir), SPLIT_RO_OUT (output json).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

#: §5.3.1 — the originally-requested absolute slice.  Reported for the record and
#: expected to be EMPTY: the S1b val fold's minimum true f_r is 1.5821, so this
#: cut selects ZERO rows (S1 had zero too).  It is reported as n=0, never as a
#: null, and no number here may be compared to the memo's -0.018 or to the v520
#: pools, which were measured on other surfaces.
VOID_SLICE_MAX_FR = 1.55
#: §5.3.2 — the substitute that is actually measurable: the bottom quintile of
#: true f_r WITHIN each cell.  This is what the optimizer does (rank inside one
#: design cell, take the flattest few) and it is the axis KILLER 2 / v520 put at
#: rho ~ 0.  Absolute cuts cannot work here because f_r LEVEL is a cell property,
#: so they select whole cells rather than the elite within them.
BAND_QUANTILE = 0.20
MIN_CELL_ROWS = 8            # a within-cell rho on <8 rows is noise, not a number
DIR = Path(os.environ.get("SPLIT_RO_DIR") or REPO / "runs" / "split_S1b")
CSV = {"champion": DIR / "rows_split_champ.csv",
       "candidate": DIR / "rows_split_cand.csv"}

#: Historical numbers, carried so a reader sees them AND sees why they are not
#: comparable (§5.3.1).  Never differenced against this arm's output.
REFERENCE_NOT_COMPARABLE = {
    "champion_family_baseline_fold_C_slice": -0.018,
    "champion_family_ci": [-0.19, 0.14],
    "v520_elite_pools_n20": [0.31, -0.31, -0.14],
    "why_not_comparable": (
        "measured on fold C / the C2 slice / 20-core elite pools, not on the S1b "
        "val fold, and on an f_r<=1.55 cut that is empty here"),
}


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _within_cell_band(true: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Mask of the bottom-``BAND_QUANTILE`` of true f_r WITHIN each cell.

    The threshold is computed from the TRUTH, so it is identical for both models
    and cannot move with a prediction -- the same reason round 2's A3 selected its
    top-K map slots by the label rather than by the model.
    """
    out = np.zeros(len(true), dtype=bool)
    for c in np.unique(cell):
        m = (cell == c) & np.isfinite(true)
        if not m.any():
            continue
        thr = np.quantile(true[m], BAND_QUANTILE)
        out |= m & (true <= thr)
    return out


def readout(df: pd.DataFrame) -> dict:
    true = pd.to_numeric(df["true_f_r"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(df["pred_f_r"], errors="coerce").to_numpy(float)
    cell = df["cell"].astype(str).to_numpy()
    ok = np.isfinite(true) & np.isfinite(pred)
    band = _within_cell_band(true, cell) & ok
    rhos, sizes = [], []
    for c in np.unique(cell[band]):
        m = band & (cell == c)
        if int(m.sum()) < MIN_CELL_ROWS:
            continue
        r = _spearman(pred[m], true[m])
        if np.isfinite(r):
            rhos.append(r)
            sizes.append(int(m.sum()))
    void = ok & (true <= VOID_SLICE_MAX_FR)
    return {
        # §5.3.2 — the measurable substitute
        "n_rows_in_band": int(band.sum()),
        "n_cells_scored": len(rhos),
        "median_within_cell_rho": float(np.median(rhos)) if rhos else float("nan"),
        "mean_within_cell_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "per_cell_rho": [round(r, 4) for r in rhos],
        "per_cell_n": sizes,
        "mae_in_band": float(np.mean(np.abs(pred[band] - true[band])))
                       if band.any() else float("nan"),
        "band_f_r_min": float(np.min(true[band])) if band.any() else float("nan"),
        "band_f_r_max": float(np.max(true[band])) if band.any() else float("nan"),
        # §5.3.1 — the void absolute slice, for the record
        "void_slice_n_rows": int(void.sum()),
        "n_rows_sub_150": int(ok.sum() and (true[ok] < 1.50).sum()),
        "min_true_f_r": float(np.min(true[ok])) if ok.any() else float("nan"),
    }


def main() -> int:
    missing = [str(p) for p in CSV.values() if not p.is_file()]
    if missing:
        print("MISSING served CSVs — serve both models on S1b first:")
        for m in missing:
            print("   ", m)
        print(__doc__.split("PREREQUISITE")[1].split("THEN")[0])
        return 2

    raw = {k: pd.read_csv(p) for k, p in CSV.items()}
    ids = {k: d["record_id"].astype(str) for k, d in raw.items()}
    if set(ids["champion"]) != set(ids["candidate"]):
        print("REFUSED: the two arms were served on different row sets")
        return 1
    # the alignment is the pairing
    base = raw["champion"].reset_index(drop=True)
    cand = (raw["candidate"].set_index(raw["candidate"]["record_id"].astype(str))
            .loc[base["record_id"].astype(str)].reset_index(drop=True))

    out = {"schema": "split_secondary_readout_v2",
           "rule_doc": "data/reports/ab2_addendum_SPLIT_20260810.md#53",
           "decides": False,
           "band": f"bottom {BAND_QUANTILE:.0%} of true f_r WITHIN each cell",
           "void_slice": f"true f_r <= {VOID_SLICE_MAX_FR} (expected n=0)",
           "min_cell_rows": MIN_CELL_ROWS,
           "reference_NOT_comparable": REFERENCE_NOT_COMPARABLE,
           "champion": readout(base),
           "candidate": readout(cand)}
    out["delta_median_rho"] = (out["candidate"]["median_within_cell_rho"]
                               - out["champion"]["median_within_cell_rho"])
    out["delta_mae"] = (out["candidate"]["mae_in_band"]
                        - out["champion"]["mae_in_band"])

    p = Path(os.environ.get("SPLIT_RO_OUT")
             or REPO / "data/reports/split_secondary_readout_20260810.json")
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    ch = out["champion"]
    print("\n=== §5.3 SECONDARY readout (decides nothing — gate-promote is the "
          "instrument) ===\n")
    print(f"§5.3.1 VOID slice  f_r <= {VOID_SLICE_MAX_FR} : "
          f"n = {ch['void_slice_n_rows']}  "
          f"(min true f_r on this surface = {ch['min_true_f_r']:.4f})")
    if ch["void_slice_n_rows"] == 0:
        print("        -> as pre-registered: EMPTY. Reported as n=0, never as a null.")
    print(f"\n§5.3.2 within-cell elite band (bottom {BAND_QUANTILE:.0%} of true f_r per cell)")
    print(f"        rows {ch['n_rows_in_band']}   cells scored (>={MIN_CELL_ROWS}) "
          f"{ch['n_cells_scored']}   band f_r "
          f"{ch['band_f_r_min']:.4f}..{ch['band_f_r_max']:.4f}\n")
    print(f"{'':12s} {'median rho':>12s} {'mean rho':>10s} {'MAE':>9s}")
    for k in ("champion", "candidate"):
        r = out[k]
        print(f"{k:12s} {r['median_within_cell_rho']:>12.4f} "
              f"{r['mean_within_cell_rho']:>10.4f} {r['mae_in_band']:>9.4f}")
    print(f"{'delta':12s} {out['delta_median_rho']:>12.4f} "
          f"{'':>10s} {out['delta_mae']:>9.4f}")
    print("\nNOT comparable to the memo's -0.018 or the v520 pools "
          f"{REFERENCE_NOT_COMPARABLE['v520_elite_pools_n20']} — "
          f"{REFERENCE_NOT_COMPARABLE['why_not_comparable']}.")
    print("\n§5.3.3 registered expectation: 68 new sub-1.50 cores = 0.13% of the")
    print("corpus, ZERO new sub-1.50 EVAL rows, and this band bottoms out at ~1.58")
    print("rather than in the sub-1.50 region the new data covers. It may well not")
    print("move and is not well positioned to detect the effect even if it is real.")
    print("A null falsifies nothing; a gain needs its own round to confirm.")
    print(f"\n  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
