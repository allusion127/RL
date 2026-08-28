"""Round 6 arm ADF — the §5.6 registered SECONDARY readout (H-family).

Pre-registered in `data/reports/ab2_addendum_ADF_20260810.md` §5.6.
**It decides nothing.** The instrument is the §5.3 frozen-surface gate.

Serves BOTH models on the four all-fresh H-family patterns that STEP 0 measured
with MASTER and compares each model's predicted Δnode_peak to the measurement:

    D1 - D2  (H2 vs H4, treated)          MASTER  +0.0072   threshold 0.005
    D3 - D4  (H1 vs H3, negative control) MASTER  +0.0005   threshold 0.005

THE PREMISE IS CORRECTED (addendum §1).  v6b ALREADY carries
`origin_adf_corner_g2`, `origin_cr1_worth` and `origin_ff_pin_max`, and H2/H4
differ on all three (+0.0245, -631.6 pcm, +0.028).  So this is NOT a
can-vs-cannot test -- both models can in principle separate H2 from H4.  It asks
whether the FACE ADFs (+0.0351 on `adf_face_g2`) and corner-g1 add anything on
top.  Three readings are registered in advance:

  * treated closer to +0.0072 than control -> the face ADFs add discrimination
  * both far off                           -> neither encoding reaches this physics
  * control already accurate                -> the v4 block sufficed, arm unnecessary

Either model firing on the negative control (|Δ(D3,D4)| > 0.005) marks the whole
readout VOID, exactly as the MASTER protocol does.

Power: n = 4 patterns, no interval, and these are 121-fresh-slot cores far
outside the equilibrium-pattern distribution both models trained on.  A
directional sanity check, not evidence.

Usage:  python 5_RL/adf_hfamily_readout.py [--treated data/models/adf_v6c]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from lpopt.data.schema import unpack_pattern                    # noqa: E402
from lpopt.model.model_api import PosValCnnBackend              # noqa: E402
from lpopt.vendor.masterrl.domain import CaseKey                # noqa: E402

FLATTEST = "b0ff11ef16de"          # fr_arms.py:34, the fixed reference pattern
LIBRARY = "ga80"
FEED = 121

#: fr_arms.py ARMS, D1-D4 (read, never imported -- fr_arms.py is off-limits to
#: modification and importing it would execute its argparse module scope).
D_ARMS = {
    "D1": ("H1_H2", {"E1": "H2", "E2": "H2"}),   # treated
    "D2": ("H3_H4", {"E1": "H4", "E2": "H4"}),   # treated
    "D3": ("H1_H2", {"E1": "H1", "E2": "H1"}),   # negative control
    "D4": ("H3_H4", {"E1": "H3", "E2": "H3"}),   # negative control
}
#: MASTER, `runs/fr_arms_d/fr_arms_results.jsonl` (memo §7).
MEASURED = {"D1": 1.4613000154, "D2": 1.4541000128,
            "D3": 1.3916000128, "D4": 1.3911000490}
THRESHOLD = 0.005                  # STEP 0 pre-registered resolution floor


def substitute(pattern, mapping):
    """fr_arms.py:129, reproduced (that file must not be imported or edited)."""
    items = []
    for it in pattern.items:
        if it.is_fresh and it.batch in mapping:
            it = dataclasses.replace(it, batch=mapping[it.batch])
        items.append(it)
    return type(pattern)(tuple(items))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="data/models/split_S1b")
    ap.add_argument("--treated", default="data/models/adf_v6c")
    ap.add_argument("--store-dir", default="data/store")
    ap.add_argument("--out", default="data/reports/adf_hfamily_readout_20260810.json")
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.store_dir) / "records.parquet")
    hit = df[df.record_id.str.startswith(FLATTEST)]
    if not len(hit):
        print(f"REFUSED: reference pattern {FLATTEST} not in the store")
        return 2
    base = unpack_pattern(hit.iloc[0]["pattern"])

    pats, cases, packed = {}, {}, {}
    for arm, (pair, mp) in D_ARMS.items():
        p = substitute(base, mp)
        pats[arm] = p
        cases[arm] = CaseKey(pair, FEED)
        packed[arm] = p.canonical()

    # -- §5.6 label hygiene: these four cores must be in NO training set ------ #
    store_patterns = set(df["pattern"].astype(str))
    leaked = {a: (packed[a] in store_patterns) for a in D_ARMS}
    print("label hygiene — constructed D-arm pattern present in the store?")
    for a, v in leaked.items():
        print(f"   {a}: {'LEAKED' if v else 'absent (good)'}")
    if any(leaked.values()):
        print("REFUSED: a D-arm core is a store row, so it may be in training.")
        return 1

    out: dict = {"schema": "adf_hfamily_readout_v1",
                 "rule_doc": "data/reports/ab2_addendum_ADF_20260810.md#56",
                 "decides": False,
                 "reference_pattern": FLATTEST,
                 "label_hygiene_all_absent_from_store": True,
                 "measured_master": {
                     "D1_minus_D2": MEASURED["D1"] - MEASURED["D2"],
                     "D3_minus_D4": MEASURED["D3"] - MEASURED["D4"],
                     "per_arm": MEASURED},
                 "threshold": THRESHOLD,
                 "models": {}}

    order = ["D1", "D2", "D3", "D4"]
    for label, d in (("control", args.control), ("treated", args.treated)):
        if not Path(d).is_dir():
            print(f"\n{label} model dir not present yet: {d}")
            out["models"][label] = {"model_dir": str(d), "status": "absent"}
            continue
        b = PosValCnnBackend.from_dir(d, store_dir=args.store_dir,
                                      library_id=LIBRARY, device="cpu")
        pk, _pks, cv, _cvs = b.predict_map_flatness(
            [pats[a] for a in order], [cases[a] for a in order])
        pk = np.asarray(pk, dtype=float)
        per = {a: float(pk[i]) for i, a in enumerate(order)}
        out["models"][label] = {
            "model_dir": str(d), "status": "ok",
            "cond_schema": b.cond_schema,
            "n_channels": int(b.encoder.n_channels),
            "node_peak": per,
            "D1_minus_D2": per["D1"] - per["D2"],
            "D3_minus_D4": per["D3"] - per["D4"],
        }

    # -- report --------------------------------------------------------------- #
    print("\n=== §5.6 H-family readout (decides nothing) ===")
    print(f"{'':10s} {'schema':>7s} {'ch':>4s} "
          f"{'D1-D2':>10s} {'|err| vs +0.0072':>17s} "
          f"{'D3-D4':>10s} {'neg-ctrl fired?':>16s}")
    m12 = out["measured_master"]["D1_minus_D2"]
    m34 = out["measured_master"]["D3_minus_D4"]
    print(f"{'MASTER':10s} {'-':>7s} {'-':>4s} {m12:>+10.4f} {'-':>17s} "
          f"{m34:>+10.4f} {'no':>16s}")
    for label in ("control", "treated"):
        r = out["models"].get(label, {})
        if r.get("status") != "ok":
            print(f"{label:10s} (not available)")
            continue
        e12 = abs(r["D1_minus_D2"] - m12)
        fired = abs(r["D3_minus_D4"]) > THRESHOLD
        print(f"{label:10s} {r['cond_schema']:>7s} {r['n_channels']:>4d} "
              f"{r['D1_minus_D2']:>+10.4f} {e12:>17.4f} "
              f"{r['D3_minus_D4']:>+10.4f} {('YES -> VOID' if fired else 'no'):>16s}")

    both = all(out["models"].get(k, {}).get("status") == "ok"
               for k in ("control", "treated"))
    if both:
        c, t = out["models"]["control"], out["models"]["treated"]
        out["treated_closer_on_D1_D2"] = bool(
            abs(t["D1_minus_D2"] - m12) < abs(c["D1_minus_D2"] - m12))
        out["void_negative_control_fired"] = bool(
            abs(c["D3_minus_D4"]) > THRESHOLD or abs(t["D3_minus_D4"]) > THRESHOLD)
        print(f"\ntreated closer to MASTER on the treated pair: "
              f"{out['treated_closer_on_D1_D2']}")
        if out["void_negative_control_fired"]:
            print("NEGATIVE CONTROL FIRED -> readout is VOID (protocol).")
    print("\nPremise correction (§1): the control ALREADY carries adf_corner_g2,")
    print("cr1_worth and ff_pin_max, and H2/H4 differ on all three. This is a")
    print("does-the-face-ADF-help test, not a can-vs-cannot test.")
    print("Power: n=4, no interval, off-distribution cores. Not evidence.")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\n  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
