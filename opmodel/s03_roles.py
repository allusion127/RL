"""Step 3: read the packaged core decks to get the REAL fresh-role split per feed."""
import re
import collections
import os
import paths as P

root = os.path.join(P.PKG, "cores")
for c in sorted(os.listdir(root)):
    f = os.path.join(root, c, "bootstrap", "MAS_INP_cy02.inp")
    if not os.path.exists(f):
        print("---", c, "no cy02 deck")
        continue
    txt = open(f, errors="replace").read()
    print("=" * 70)
    print(c)
    for tag in ("%LPD_SHF", "%GEN_DIM", "%OPR_OPT", "%DEP_OPT"):
        m = re.search(re.escape(tag) + r"(.*?)(?=\n\s*%|\Z)", txt, re.S)
        if not m:
            continue
        blk = m.group(1)
        if tag == "%LPD_SHF":
            toks = blk.split()
            cnt = collections.Counter(toks)
            print("  LPD_SHF n_tokens=%d" % len(toks))
            print("   ", dict(sorted(cnt.items(), key=lambda kv: -kv[1])))
        else:
            print("  " + tag + ":", " | ".join(
                ln.strip() for ln in blk.strip().splitlines()[:4]))
