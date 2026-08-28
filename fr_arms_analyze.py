"""Score the fixed-loading-pattern fuel-swap experiment against its falsifiers.

Reads every ``runs/fr_arms*/fr_arms_results.jsonl`` and reports, per reference
anchor, the measured transfer of the lattice pin-power form factor FF into core
F_r.  The pre-registered falsifiers live in
``data/reports/flat_assembly_fr_plan_20260802.md`` §3.3 and are evaluated here
verbatim so the verdict cannot drift with the telling.

FF values are the exact DeCART truth: max of the 16x16 %DIST pin-power map at
BU=0, read from ``FEASIBLE_PACKAGE/hgc/FA_<type>.HGC``.  They are not surrogate
predictions -- the surrogate was trained to emulate these very maps, so using it
here would substitute a +-0.002 approximation for an exact number.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

# type -> FF (pin-power form factor at BU=0).  Verified three ways: raw HGC
# re-parse, scratchpad ff_lib.csv, and data/store/fuel_types.parquet:ff_pin_max
# agree to 0.000000 on all 36 recoverable ga80 types.
FF = {"E1": 1.146, "E2": 1.152, "E3": 1.101, "E4": 1.139, "A8": 1.157,
      "A2": 1.178, "H3": 1.117, "H4": 1.171, "K5": 1.118, "K6": 1.149,
      "L3": 1.115, "L4": 1.145, "J1": 1.113, "J2": 1.146,
      # paramA (different library -- reported, never pooled into a ga80 fit)
      "Q1": 1.122, "Q2": 1.174, "Q7": 1.205, "Q8": 1.209}

# arm -> (E1-role type, E2-role type).  The E1 role fills the 68 low-power
# fresh slots, the E2 role the 53 hot ones (flat anchor); 64/57 at the minfr
# anchor.  C5/C6 put ONE type in both roles.
ROLES = {"A0": ("E1", "E2"), "A1": ("E3", "E4"), "A2": ("A8", "A2"),
         "B0": ("Q1", "Q2"), "B1": ("Q7", "Q8"),
         "C1": ("H3", "H4"), "C2": ("K5", "K6"), "C3": ("L3", "L4"),
         "C4": ("J1", "J2"), "C5": ("E3", "E3"), "C6": ("A2", "A2")}

PARAMA = {"B0", "B1"}
HOMOGENEOUS = {"C5", "C6"}          # single type in all fresh slots

# Nodal power of the hottest fresh slot of each role, measured from maps.npz for
# the reference core, and the in-core amplification A that closes the
# decomposition F_r = A * max(p_boc * FF) on the control.
ANCHOR = {"flat":  {"p_e1": 1.1934, "p_e2": 1.2080, "A": 1.0928,
                    "f_r": 1.5207, "node_peak": 1.2085, "record": "b0ff11ef16de"},
          "minfr": {"p_e1": None, "p_e2": None, "A": None,
                    "f_r": 1.4636, "node_peak": 1.2620, "record": "deb058c00433"}}


def load() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob("runs/fr_arms*/fr_arms_results.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") != "converged":
                continue
            ref = r.get("reference") or "flat"
            # later sweeps supersede earlier ones (the first flat sweep lost the
            # flatness scalars to a NaN-max bug and was re-run)
            prev = out[ref].get(r["arm"])
            if prev is None or np.isnan(prev.get("node_peak", np.nan)):
                out[ref][r["arm"]] = r
    return out


def fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    x, y = np.asarray(xs), np.asarray(ys)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    sl, ic = np.polyfit(x, y, 1)
    pred = sl * x + ic
    ss = 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-15)
    return float(sl), float(ic), float(ss)


def main() -> None:
    data = load()
    for ref in ("flat", "minfr"):
        arms = data.get(ref)
        if not arms:
            continue
        a = ANCHOR[ref]
        print(f"\n{'='*104}\nANCHOR '{ref}'  record {a['record']}  "
              f"stored F_r {a['f_r']:.4f}  node_peak {a['node_peak']:.4f}\n{'='*104}")
        print(f"{'arm':4}{'pair':10}{'FF_e1':>7}{'FF_hot':>7}{'F_r':>9}{'dF_r':>8}"
              f"{'node_peak':>10}{'dpeak':>8}{'map_cov':>9}{'dcov':>9}"
              f"{'cyclen':>9}{'CBC':>8}{'F_q':>8}")
        ctrl = arms.get("A0")
        xs_ga, ys_ga, homo = [], [], []
        for name in sorted(arms):
            r = arms[name]
            fom = r["fom"]
            e1, e2 = ROLES[name]
            ffh = FF[e2]
            d_fr = fom["F_r"] - ctrl["fom"]["F_r"] if ctrl else float("nan")
            pk = r.get("node_peak", float("nan"))
            cv = r.get("map_cov", float("nan"))
            d_pk = pk - ctrl.get("node_peak", float("nan")) if ctrl else float("nan")
            d_cv = cv - ctrl.get("map_cov", float("nan")) if ctrl else float("nan")
            tag = "*" if name in PARAMA else ("#" if name in HOMOGENEOUS else " ")
            print(f"{name+tag:4}{r['pair']:10}{FF[e1]:7.3f}{ffh:7.3f}{fom['F_r']:9.4f}"
                  f"{d_fr:+8.4f}{pk:10.4f}{d_pk:+8.4f}{cv:9.5f}{d_cv:+9.5f}"
                  f"{fom['cyclen']:9.3f}{fom['CBC_max']:8.1f}{fom['F_q']:8.4f}")
            if name in PARAMA:
                continue
            if name in HOMOGENEOUS:
                homo.append((ffh, fom["F_r"]))
            else:
                xs_ga.append(ffh)
                ys_ga.append(fom["F_r"])
        print("  * paramA library (different package) -- reported, never pooled")
        print("  # one type in ALL fresh slots -- its own series, not pooled with the rest")
        if len(xs_ga) >= 2:
            sl, ic, r2 = fit(xs_ga, ys_ga)
            print(f"\n  ga80 natural-order fit: F_r = {sl:.4f}*FF_hot {ic:+.4f}   "
                  f"R2={r2:.4f}   n={len(xs_ga)}")
            if ref == "flat":
                print(f"    separable model predicts slope {a['A']*a['p_e2']:.4f}; "
                      f"attenuation = {sl/(a['A']*a['p_e2']):.3f}")
                print(f"    F1 falsifier (slope < 0.66): "
                      f"{'FIRES' if sl < 0.66 else 'does not fire'}")
                print(f"    confirmation band [1.0,1.6]: "
                      f"{'inside' if 1.0 <= sl <= 1.6 else 'OUTSIDE'}")
        if len(homo) >= 2:
            sl, _ic, _r2 = fit([h[0] for h in homo], [h[1] for h in homo])
            print(f"  homogeneous-load 2-point slope: {sl:.4f}  "
                  f"(FF {homo[0][0]:.3f}->{homo[-1][0]:.3f}, "
                  f"F_r {homo[0][1]:.4f}->{homo[-1][1]:.4f})")
        if ctrl and ref == "flat":
            amps = []
            for name, r in arms.items():
                if name in PARAMA or name in HOMOGENEOUS:
                    continue
                e1, e2 = ROLES[name]
                amps.append(r["fom"]["F_r"] / max(a["p_e1"] * FF[e1], a["p_e2"] * FF[e2]))
            print(f"  F5 amplification A across ga80 arms: {min(amps):.5f}..{max(amps):.5f} "
                  f"(spread {max(amps)-min(amps):.5f}; F5 fires above 0.03) -> "
                  f"{'FIRES' if max(amps)-min(amps) > 0.03 else 'does not fire'}")


if __name__ == "__main__":
    main()
