"""Audit / invalidate / regenerate the flatness program's judging split.

::

    python -m lpopt.tools.audit_c2_split                # measure and report
    python -m lpopt.tools.audit_c2_split --invalidate   # stamp the verdict INTO S2.json
    python -m lpopt.tools.audit_c2_split --regenerate   # build the section 7.2 split
    python -m lpopt.tools.audit_c2_split --show-c2      # build C2 and print its provenance

Why a tool and not a warning at import time
-------------------------------------------
``data/splits/S2.json`` currently holds the LEGACY leave-pair-out manifest
(``kind="leave_pair"``, holdout pairs ``C3_C6`` / ``A01_B05``, written
2026-07-18).  It loads, it validates, every one of its ids resolves in the
store, and its complement fold computes without error -- it just is not the
split section 7.1 asks for, and its complement is "everything added to the store
since July 18" rather than a designed holdout.

A stale artifact that looks healthy is more dangerous than a missing one, so the
staleness is made *material*: :func:`--invalidate <main>` writes the measured
audit into the manifest itself and flips ``status`` off ``ok``, while
``train_ids``/``val_ids`` are left byte-identical so existing readers keep
working.  Nothing here guesses; every number in the audit is measured against
the store at the moment it runs.

``--regenerate`` writes the section 7.2 cell-level holdout under a NEW name by
default (``S2_flat``) rather than overwriting ``S2.json``: the legacy S2 is still
referenced by :mod:`lpopt.model.evaluate` as the leave-pair-out functional check,
and silently repurposing a split name is how this situation arose the first
time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..data.store import StoreReader
from ..model.c2_slice import (
    FLATNESS_PROGRAM, MAX_UNCOVERED_FRAC, SplitStaleError, audit_split, build_c2,
    make_flat_cell_split, mark_stale, render_audit, write_audit,
)
from ..model.splits import SplitManifest

DEFAULT_STORE = "data/store"
DEFAULT_SPLITS = "data/splits"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lpopt.tools.audit_c2_split")
    ap.add_argument("--store-dir", default=DEFAULT_STORE)
    ap.add_argument("--splits-dir", default=DEFAULT_SPLITS)
    ap.add_argument("--split", default="S2",
                    help="the manifest the flatness A/B would read")
    ap.add_argument("--audit-out", default=None,
                    help="sidecar path (default <splits-dir>/<split>.audit.json)")
    ap.add_argument("--invalidate", action="store_true",
                    help="stamp status=stale + the audit INTO the manifest "
                         "(train_ids/val_ids untouched)")
    ap.add_argument("--regenerate", action="store_true",
                    help="build the section 7.2 cell-holdout split")
    ap.add_argument("--regen-name", default="S2_flat",
                    help="name/file for the regenerated split (NOT S2 by default: "
                         "S2 is still the leave-pair-out functional check)")
    ap.add_argument("--e-core-holdout-from", type=float, default=5.75,
                    help="cells at or above this median e_core become the "
                         "extrapolation stratum (section 7.2)")
    ap.add_argument("--cell-val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-uncovered-frac", type=float, default=MAX_UNCOVERED_FRAC)
    ap.add_argument("--show-c2", action="store_true",
                    help="build the C2 slice and print its provenance block")
    args = ap.parse_args(argv)

    reader = StoreReader(args.store_dir)
    df = reader.records
    split_path = Path(args.splits_dir) / f"{args.split}.json"
    if not split_path.exists():
        print(f"no split at {split_path}")
        return 1
    manifest = SplitManifest.from_json(split_path)

    audit = audit_split(manifest, df, program=FLATNESS_PROGRAM,
                        max_uncovered_frac=args.max_uncovered_frac)
    print(render_audit(audit))

    out = Path(args.audit_out or (split_path.with_suffix(".audit.json")))
    write_audit(audit, out)
    print(f"-> {out}")

    if args.invalidate and audit["stale"]:
        mark_stale(split_path, audit)
        print(f"-> {split_path} status=stale (ids untouched)")
    elif args.invalidate:
        print("not invalidating: the audit says this split is fresh")

    if args.regenerate:
        m = make_flat_cell_split(
            df, name=args.regen_name, seed=args.seed,
            e_core_holdout_from=args.e_core_holdout_from,
            cell_val_frac=args.cell_val_frac)
        p = Path(args.splits_dir) / f"{args.regen_name}.json"
        m.to_json(p)
        g = m.groups
        print(f"-> {p}  kind={m.kind} train={m.n_train} val={m.n_val} "
              f"in_domain_val_cells={len(g['in_domain_val_cells'])} "
              f"extrapolation_cells={len(g['extrapolation_cells'])} "
              f"of {g['n_cells_total']}")
        manifest, split_path = m, p

    if args.show_c2:
        try:
            c2 = build_c2(df, manifest)
        except SplitStaleError as exc:
            print(f"\nC2 REFUSED:\n{exc}")
            return 2
        print("\n" + json.dumps(c2.provenance, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
