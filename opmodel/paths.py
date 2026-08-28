"""Shared paths + DeCART k(BU) loader for the operating-point re-screen.

Read-only on everything under 5_RL/data.
"""
import re
import json
import numpy as np

RL = ("C:\\Users\\USER\\Desktop\\CT&RPL\\2_Project\\KNF_LEU+ "
      "\uc2dc\ubc94\uc5f0\ub8cc\ubd09 \uc7a5\uc804 \uc778\ud5c8\uac00 \uacfc\uc81c"
      "\\2026\\2_\uacc4\uc0b0\\5_RL")
PKG = RL + "\\data\\design\\package"
HGC = PKG + "\\hgc"
SCR = ("C:\\Users\\USER\\AppData\\Local\\Temp\\claude"
       "\\c--Users-USER-Desktop-CT-RPL-2-Project-KNF-LEU------------------2026-2---"
       "\\8888f052-fa4d-46f0-a439-ef3441b3b061\\scratchpad")
LAT = SCR + "\\lat1600"
V2 = SCR + "\\lat1600_v2"

# ---- core physics constants (lpopt.coredeck.CoreParams) --------------------
POWER, HM = 3983.0, 104.8
RATE = POWER / HM / 1000.0          # MWd/kgHM per EFPD  = 0.0380058
NSLOT = 241
# cy1 all-fresh full-core census of the two fresh aliases (coredeck QUARTER_BCH_MAP)
W_CY1 = (129, 112)
# equilibrium fresh-feed role split at feed 121 (designs.json lat1600_role)
ROLE121 = (68, 53)                  # (E1-role, E2-role hot)

PAT = re.compile(r"^\s*([0-9.]+)\s+MWD/KGHM\s+[0-9.]+\s+EFPD\s+K-CONV\s*=\s*([0-9.]+)")


def load_decart(names=None):
    """alias -> (bu, kconv) from the DeCART lattice outputs MASTER's library was
    built from.  These are the ground truth the validated model uses."""
    import glob
    import os
    if names is None:
        names = sorted(os.path.basename(p)[3:-4]
                       for p in glob.glob(HGC + "\\FA_*.out"))
    out = {}
    for n in names:
        b, k = [], []
        with open(f"{HGC}\\FA_{n}.out", errors="replace") as fh:
            for ln in fh:
                m = PAT.match(ln)
                if m:
                    b.append(float(m.group(1)))
                    k.append(float(m.group(2)))
        if not b:
            continue
        b = np.array(b)
        _, i = np.unique(b, return_index=True)
        i.sort()
        out[n] = (b[i], np.array(k)[i])
    return out


def load_designs():
    """alias -> design record from the package designs.json."""
    d = json.load(open(PKG + "\\designs.json", encoding="utf-8"))
    return {r["alias"]: r for r in d["designs"]}
