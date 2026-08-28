"""P3a calibration cell — the ONE cell where a 3-type delta has been MEASURED.

``hgd569 / f125`` (paramA ``P6253Z1G06N24_P6253Z2G10N24``, e_core 5.6944) is the
only place in the programme where a 2-type campaign, a 2-type seeding control
and a 3-type campaign have all run at the same cell, feed, budget, objective and
gate set.  The measured answer is:

    2-type joint-clean F_r  1.6357          (fpcamp_minfr_hgd569_f125)
    3-type joint-clean F_r  1.5993          (fpcamp_minfr_triple_f125)
    measured delta          -0.0364
    (raw-vs-raw             1.6088 -> 1.5956 = -0.0132)

The sweep never visits this cell — the v3 mesh pick at e5.7 is a different
(cross-spec) pair — so the sweep's own claim about grading has exactly one
falsifiable check available, and this script is it.  It runs the sweep's
``run_case`` verbatim on the hgd569 pair and on the triple the campaign actually
used, and prints predicted-vs-measured side by side.

Registered before it was run (PREREG sec. 6): if the sign does not reproduce, the
readout says so in the headline and no confidence interval is claimed for the
anchor predictions.

    python calib_multitype_hgd569.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

OUT = BASE / "data" / "reports" / "mesh_multitype_20260818"

#: the cell, and the two cases the campaigns actually ran
LIB = "paramA"
PAIR = "P6253Z1G06N24_P6253Z2G10N24"
TRIPLE = "P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24"
FEED = 125
E_CORE = 5.694438                    # the pair's 50/50 value, per the deck header

#: measured marks (tripletype_f125_results_20260817.md sec. 1-2)
MEASURED = {
    "2type_clean_f_r": 1.6357, "2type_raw_f_r": 1.6088,
    "3type_clean_f_r": 1.5993, "3type_raw_f_r": 1.5956,
    "2type_clean_pin": 76.955, "3type_clean_pin": 75.526,
    "2type_clean_cbc": 1565.46, "3type_clean_cbc": 1597.33,
    "2type_clean_cyclen": 730.85, "3type_clean_cyclen": 730.5,
}


def main() -> int:
    import torch
    torch.set_num_threads(8)
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.model.model_api import PosValCnnBackend

    from mesh_multitype import P_TH_MW, run_case
    from scoping_mesh import N_ELITE_DONOR, _store_feasible, pair_hm_tu

    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / "calibration_hgd569.log"
    logf.write_text("", encoding="utf-8")

    def log(m: str = "") -> None:
        print(m, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(str(m) + "\n")

    cfg = load_config(str(BASE / "lpopt.inp"))
    fuel_df = pd.read_parquet(BASE / "data/store/fuel_types.parquet")
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    store = store[(store.valid == True) & (store.converged == True)]   # noqa: E712
    donors = store[_store_feasible(store)].nsmallest(N_ELITE_DONOR * 2, "f_r")
    _per_fa, m_hm = pair_hm_tu(fuel_df, LIB, *PAIR.split("_"))

    model = PosValCnnBackend.from_dir(BASE / "data/models/s1i",
                                      store_dir=BASE / "data/store",
                                      library_id="ga80", device="cpu")
    model.quantile_targets = ()

    log("=== P3a calibration cell — hgd569 / f125 (PREREG §6) ===")
    log(f"pair   {PAIR}   e_core {E_CORE}")
    log(f"triple {TRIPLE}")
    log("")
    t0 = time.time()
    res = {}
    for k, case in ((2, PAIR), (3, TRIPLE)):
        r, _front = run_case(model, cfg, store, donors, LIB, case,
                             case.split("_"), E_CORE, FEED, k, log)
        res[k] = r

    d_pred = res[3]["min_f_r_clean"] - res[2]["min_f_r_clean"]
    d_meas = MEASURED["3type_clean_f_r"] - MEASURED["2type_clean_f_r"]
    log("")
    log("--- predicted vs MEASURED, joint-clean (all axes but F_r) ---")
    log(f"{'':22s} {'2-type':>10s} {'3-type':>10s} {'delta':>10s}")
    log(f"{'PREDICTED F_r floor':22s} {res[2]['min_f_r_clean']:10.4f} "
        f"{res[3]['min_f_r_clean']:10.4f} {d_pred:+10.4f}")
    log(f"{'MEASURED  F_r':22s} {MEASURED['2type_clean_f_r']:10.4f} "
        f"{MEASURED['3type_clean_f_r']:10.4f} {d_meas:+10.4f}")
    log(f"{'PREDICTED pin BU':22s} {res[2]['clean_pin']:10.3f} "
        f"{res[3]['clean_pin']:10.3f} "
        f"{res[3]['clean_pin'] - res[2]['clean_pin']:+10.3f}")
    log(f"{'MEASURED  pin BU':22s} {MEASURED['2type_clean_pin']:10.3f} "
        f"{MEASURED['3type_clean_pin']:10.3f} "
        f"{MEASURED['3type_clean_pin'] - MEASURED['2type_clean_pin']:+10.3f}")
    log("")
    sign_ok = bool(np.sign(d_pred) == np.sign(d_meas))
    log(f"SIGN REPRODUCED: {sign_ok}   "
        f"(predicted {d_pred:+.4f} vs measured {d_meas:+.4f}; "
        f"magnitude ratio {d_pred / d_meas if d_meas else float('nan'):.2f})")
    if not sign_ok:
        log("REGISTERED CONSEQUENCE (PREREG §6): the sweep does not reproduce the "
            "one measured grading delta.  No confidence is claimed for the anchor "
            "predictions, and this line goes in the readout headline.")
    log(f"\n{time.time() - t0:.0f}s")

    json.dump(dict(cell="hgd569_f125", pair=PAIR, triple=TRIPLE, feed=FEED,
                   e_core=E_CORE, m_hm_tU=m_hm,
                   predicted={str(k): {kk: (None if isinstance(vv, float)
                                            and not np.isfinite(vv) else vv)
                                       for kk, vv in v.items()
                                       if not isinstance(vv, (list, dict))}
                              for k, v in res.items()},
                   measured=MEASURED, delta_predicted=d_pred,
                   delta_measured=d_meas, sign_reproduced=sign_ok),
              open(OUT / "calibration_hgd569.json", "w", encoding="utf-8"),
              indent=1, default=float)
    log(f"wrote {OUT / 'calibration_hgd569.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
