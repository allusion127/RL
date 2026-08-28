"""Step 5: harvest k_inf(BU) and FF(BU) from the REFERENCE states of every HGC in
BOTH packages (ga80 FEASIBLE_PACKAGE + paramA design package).

Same parser convention as scratchpad/ff_harvest.py (which produced ff_lib.json):
REFERENCE %TITL blocks, kinf from the 2nd numeric header line, FF = max of the
first 16x16 %DIST map.  Cached to hgc_curves.npz so later steps are instant.
"""
import re
import json
import glob
import os
import numpy as np

ROOT = ("C:\\Users\\USER\\Desktop\\CT&RPL\\2_Project\\KNF_LEU+ "
        "\uc2dc\ubc94\uc5f0\ub8cc\ubd09 \uc7a5\uc804 \uc778\ud5c8\uac00 \uacfc\uc81c"
        "\\2026\\2_\uacc4\uc0b0")
PKGS = {"ga80": ROOT + "\\3_GA_Surrogate\\FEASIBLE_PACKAGE\\hgc",
        "paramA": ROOT + "\\5_RL\\data\\design\\package\\hgc"}
OUT = os.path.dirname(os.path.abspath(__file__))

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def floats(s):
    out = []
    for t in FLOAT_RE.findall(s):
        try:
            out.append(float(t.replace("D", "E").replace("d", "e")))
        except ValueError:
            pass
    return out


def first_dist_map(lines):
    di = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("%DIST"):
            di = i
            break
    if di is None:
        return None
    rows = []
    for ln in lines[di + 1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("%"):
            break
        try:
            v = [float(t) for t in s.split()]
        except ValueError:
            break
        if len(v) != 16:
            break
        rows.append(v)
        if len(rows) >= 16:
            break
    return np.array(rows) if len(rows) == 16 else None


def parse_hgc(path):
    text = open(path, errors="replace").read()
    recs = []
    for blk in re.split(r"(?m)^%TITL", text):
        if not blk.strip():
            continue
        bl = blk.splitlines()
        case = None
        hdr = []
        for i, ln in enumerate(bl):
            if "CASE ::" in ln:
                case = ln.split("CASE ::")[1].strip()
                hdr = bl[i + 1:i + 5]
                break
        if case is None or not case.upper().startswith("REFERENCE"):
            continue
        try:
            la = floats(hdr[1])
            bu, kinf = la[1], la[2]
        except Exception:
            continue
        m = first_dist_map(bl)
        if m is None:
            continue
        recs.append((bu, kinf, float(m.max())))
    recs.sort()
    a = np.array(recs)
    return a[:, 0], a[:, 1], a[:, 2]


if __name__ == "__main__":
    store = {}
    for lib, d in PKGS.items():
        for f in sorted(glob.glob(d + "\\FA_*.HGC")):
            tid = os.path.basename(f)[3:-4]
            try:
                bu, k, ff = parse_hgc(f)
            except Exception as e:
                print("SKIP", lib, tid, e)
                continue
            store[f"{lib}:{tid}:bu"] = bu
            store[f"{lib}:{tid}:k"] = k
            store[f"{lib}:{tid}:ff"] = ff
    np.savez_compressed(OUT + "\\hgc_curves.npz", **store)
    keys = sorted({k.rsplit(":", 1)[0] for k in store})
    print("harvested %d types" % len(keys))
    print(", ".join(keys))

    # cross-check: paramA .out K-CONV vs HGC kinf for the four new types
    import paths as P
    dec = P.load_decart(["T3", "T4", "T5", "T6"])
    print("\n=== HGC kinf vs .out K-CONV (paramA T3-T6) ===")
    for n in ("T3", "T4", "T5", "T6"):
        b, k = store[f"paramA:{n}:bu"], store[f"paramA:{n}:k"]
        db, dk = dec[n]
        d = [1e5 * ((1 - 1 / np.interp(x, b, k)) - (1 - 1 / np.interp(x, db, dk)))
             for x in (0.5, 10, 20, 30, 40, 50)]
        print(f"  {n}: n_state={len(b)} bu {b.min():.1f}..{b.max():.1f} "
              f"drho(pcm) at 0.5/10/20/30/40/50 = " + " ".join(f"{x:+.0f}" for x in d))
    print("\n=== ff_boc from HGC vs ff_lib.csv ===")
    import csv
    P2 = P.SCR + "\\ff_lib.csv"
    with open(P2) as fh:
        for r in csv.DictReader(fh):
            key = f"{r['lib']}:{r['type_id']}:ff"
            if key in store:
                got = store[key][0]
                if abs(got - float(r["ff_boc"])) > 0.0015:
                    print("  MISMATCH", key, got, r["ff_boc"])
    print("  (only mismatches > 0.0015 printed)")
