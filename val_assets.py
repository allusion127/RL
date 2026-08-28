"""Coverage-fill preflight: resolve MASTER assets for a deck's strata WITHOUT
running MASTER.  Proves each stratum's (pair, feed) can stage a restart + template
deck, which is the one thing `produce --dry-run` (StubEvaluator) does NOT check.

Usage: python val_assets.py <fill_deck.inp> [max_strata]
"""
import sys
from pathlib import Path
from lpopt.config import load_config
from lpopt.data.fuel_types import FuelLibrary
from lpopt.search.resolver import build_case_resolver   # per-library routing (paramA!)
from lpopt.vendor.masterrl.dataset import CaseKey

deck = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 99
cfg = load_config(deck)
root = Path(deck).resolve().parent
fuel = FuelLibrary.from_parquet(root / "data/store/fuel_types.parquet")
ok = bad = 0
for st in cfg.produce.strata[:limit]:
    pair = st.pairs[0]
    # produce builds ONE resolver per stratum library -> mirror that exactly.
    res = build_case_resolver(cfg, fuel, st.library)
    try:
        a = res.resolve(CaseKey(pair=pair, feed=st.feed))
        good = a.restart_path is not None
        ok += int(good); bad += int(not good)
        print(f"  {'OK  ' if good else 'FAIL'} {st.name:22s} lib={st.library:7s} "
              f"kind={a.kind} lvl={a.fallback_level} prov={a.restart_provenance} "
              f"rst={'Y' if a.restart_path else 'N'} tpl={'Y' if a.template_deck_path else 'N'}")
        if not good:
            for n in a.notes[:2]:
                print(f"        note: {str(n)[:110]}")
    except Exception as e:
        bad += 1
        print(f"  FAIL {st.name:22s} {type(e).__name__}: {str(e)[:100]}")
print(f"RESULT resolved={ok} failed={bad}")
