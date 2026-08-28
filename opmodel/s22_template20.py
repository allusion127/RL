"""Step 22: what is the FROZEN Gd 1/8 layout in the IGD_20 base templates, and
which base deck serves each recommended new design?"""
import glob
import os
import sys
import paths as P

sys.path.insert(0, P.LAT)
from author_decks import find_map_block, FIXED, ZONING_PB  # noqa: E402

APR = os.path.dirname(P.RL) + "\\0_APR1400"
for fam in ("5.8_5.1", "260624"):
    for ng in (16, 20, 24):
        for gw in (6, 8, 10):
            d = f"{APR}\\{fam}\\FA\\IGD_{ng}\\{gw}_{ng}_z1"
            hits = sorted(glob.glob(d + "\\dec_FA_*.inp"))
            if not hits:
                continue
            lines = open(hits[0], errors="replace").read().splitlines()
            try:
                _, rows = find_map_block(lines)
            except SystemExit as e:
                print(f"  {fam}/IGD_{ng}/{gw}_{ng}_z1: {e}")
                continue
            gd = [(r, c) for r in range(8) for c in range(r + 1)
                  if rows[r][c] == 3]
            zon = {(r, c) for r in range(8) for c in range(r + 1)
                   if rows[r][c] == 2}
            lay = ";".join(f"{r}:{c}" for r, c in gd)
            print(f"  {fam:>8}/IGD_{ng}/{gw:>2}_{ng}_z1  "
                  f"{os.path.basename(hits[0]):>16}  layout {lay:<26} "
                  f"zoning{'=PB' if zon == ZONING_PB else '!=PB'}")
