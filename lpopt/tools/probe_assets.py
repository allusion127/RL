"""Resolve every (pair, feed) a deck will stage -- the exact precondition
verify.py:876 enforces -- WITHOUT running MASTER and without staging anything.

`lpopt produce --dry-run` cannot answer this: dry runs set stage_decks=False, so
the MissingCaseAssetError guard is skipped entirely and every chain "passes".
Run this before committing chains to a deck whose pairs have not been produced
on this box before.

Usage:  python probe_assets.py <deck.inp>
Exit 0 = every stratum resolvable; exit 1 = at least one is not.
"""
import sys
from pathlib import Path

from lpopt.config import load_config
from lpopt.data.fuel_types import FuelLibrary, case_e_core
from lpopt.search.resolver import build_case_resolver
from lpopt.vendor.masterrl.domain import CaseKey

deck = Path(sys.argv[1])
cfg = load_config(deck)

# Loaded from the store sidecar, the same source produce uses.  NOT optional:
# without it every e_core reads NULL and the probe reports a false alarm on a
# perfectly good deck (observed 2026-07-27).  Resolution itself does not need it,
# so a hard failure here is better than a silent None.
_ft = Path(cfg.model.store_dir) / "fuel_types.parquet"
if not _ft.is_file():
    _ft = deck.parent / cfg.model.store_dir / "fuel_types.parquet"
fuel = FuelLibrary.from_parquet(_ft)

# One resolver per library actually named by the strata -- paramA and ga80 route
# to different packages, so a single resolver would answer for the wrong one.
resolvers = {}


def _resolver_for(lib):
    if lib not in resolvers:
        resolvers[lib] = build_case_resolver(cfg, fuel, library_id=lib)
    return resolvers[lib]


def _e_core(pair):
    """`e_core` exactly as produce derives it (produce.py:648-658).

    Checked here because the failure is SILENT: `_enrichment` catches KeyError and
    returns (None, None), so a deck whose pair names are in the wrong name space
    (short alias instead of type_id, for paramA) stages and runs perfectly and
    writes rows with a NULL e_core -- useless for v6 conditioning and inconsistent
    with the cell's existing rows.  Assets resolving is NOT enough.

    Goes through `case_e_core` (NOT `partition("_")`), so a graded 3..5-type case
    id reports its composition mean instead of a false NULL alarm -- a two-member
    case still takes the identical `pair_e_core(a, b, 0.5)` call.
    """
    if fuel is None:
        return None
    members = [p for p in pair.split("_") if p]
    if len(members) < 2:
        return None
    try:
        return float(case_e_core(fuel, members, st.library))
    except Exception:
        return None


bad = 0
print(f"{'stratum':26s} {'pair':30s} {'feed':>4s}  {'rst':3s} {'tpl':3s} {'e_core':>7s}  provenance")
for st in cfg.produce.strata:
    for pair in st.pairs:
        try:
            r = _resolver_for(st.library).resolve(CaseKey(pair, int(st.feed)))
        except Exception as exc:                       # AssetResolutionError etc.
            print(f"{st.name:26s} {pair:30s} {st.feed:4d}  RAISED {type(exc).__name__}: {exc}")
            bad += 1
            continue
        rst = "Y" if r.restart_path is not None else "N"
        tpl = "Y" if r.template_deck_path is not None else "N"
        ec = _e_core(pair)
        if rst == "N" or tpl == "N" or ec is None:
            bad += 1
        ecs = f"{ec:7.3f}" if ec is not None else "   NULL"
        print(f"{st.name:26s} {pair:30s} {st.feed:4d}  {rst:3s} {tpl:3s} {ecs}  "
              f"{r.restart_provenance}")

print(f"\nRESULT: {'ALL USABLE' if not bad else str(bad) + ' UNUSABLE (missing asset or NULL e_core)'}")
sys.exit(1 if bad else 0)
