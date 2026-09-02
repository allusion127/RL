"""Corpus preparation for policy net v2.

Two one-off jobs, both append-only and both backed up before they write:

``backfill``
    Add ``parent_/child_/d_fresh_enr_mass`` to the EXISTING ``steps.parquet``
    rows.  ``mine_policy_corpus.PHYSICS`` now carries ``fresh_enr_mass`` (the
    reactivity covariate the ablation post-mortem prescribed), so freshly mined
    rows already have the three columns and the historical 28,063 do not.  The
    values are recomputed from the ``parent_pattern`` / ``child_pattern`` strings
    the corpus already stores, with ``mine_policy_corpus.ring_profile`` itself —
    no second implementation, and no read of ``data/store`` at all.

``backfill-fxy``
    Add ``parent_f_xy`` / ``child_f_xy`` / ``d_f_xy`` / ``improved_fxy`` and
    ``burn_state`` to the EXISTING ``steps.parquet`` rows
    (``intervention_wave_r1_results_20260830.md`` s11-3: the F_xy-era waves make
    F_xy their PRIMARY response and the 80-column schema had nowhere to put the
    label, so the wave's own answers never reached the corpus).  The F_xy values
    are JOINED from ``data/store/records.parquet`` on ``record_id`` - they are
    MASTER readings and are not recomputable - while ``burn_state`` is recovered
    from the ``parent_pattern`` / ``child_pattern`` strings the corpus already
    stores, with ``mine_policy_corpus.move_burn_state`` itself.  ADD-ONLY: no
    existing column is read back, rewritten or reordered.  Dry-run by default;
    ``--apply`` writes, after a backup.

``verify``
    Re-mine one campaign that is already in the corpus and assert every column
    matches, which is the check ``ablation_analyze`` used to license its appender.

Mining a NEW campaign is NOT done here: ``ablation_analyze.py corpus --campaign
<name> --lineage <tag>`` is the registered appender and is reused verbatim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

STEPS = BASE / "data/policy/steps.parquet"
FUEL_TYPES = BASE / "data/store/fuel_types.parquet"
STORE = BASE / "data/store/records.parquet"

#: Where the F_xy-column migration parks its pre-write copy.  On the data drive,
#: beside the wave backups, because a 9 MB sibling of the file being migrated is
#: exactly the copy a mistaken `git clean` takes with it.
FXY_BACKUP = Path("E:/lpopt_data/5_RL/backups/"
                  "steps.parquet.bak_pre_fxy_cols_20260830")

#: The three columns ``PHYSICS`` gained.  Inserted in ``build_steps`` order,
#: i.e. immediately after the ``*_fresh_enr_r_center`` triple, so a backfilled
#: frame and a freshly mined one have the SAME column order and the appender's
#: schema-drift guard stays meaningful.
NEW_COLUMNS = ("parent_fresh_enr_mass", "child_fresh_enr_mass", "d_fresh_enr_mass")
_ANCHOR = "d_fresh_enr_r_center"


def _mass_column(patterns: pd.Series, libs: pd.Series,
                 enrichment: dict[str, dict[str, float]]) -> np.ndarray:
    """``fresh_enr_mass`` per row, memoized on (pattern, library_id).

    ``build_steps`` evaluates BOTH sides of an edge under the CHILD's library
    (``enr = enrichment.get(str(lib))`` with ``lib = children['library_id']``),
    so the caller passes the step's own ``library_id`` for the parent side too.
    """
    import mine_policy_corpus as M

    cache: dict[tuple[str, str], float] = {}
    out = np.empty(len(patterns), dtype=float)
    for i, (pat, lib) in enumerate(zip(patterns, libs, strict=True)):
        key = (pat, lib)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = M.ring_profile(
                pat, enrichment.get(str(lib)))["fresh_enr_mass"]
        out[i] = hit
    return out


def cmd_backfill(args) -> int:
    import mine_policy_corpus as M

    steps = pd.read_parquet(args.steps)
    print(f"[backfill] {len(steps)} rows, {len(steps.columns)} columns")
    already = [c for c in NEW_COLUMNS if c in steps.columns]
    if already and not args.force:
        print(f"[backfill] already present: {already} - nothing to do")
        return 0

    enrichment = M.load_enrichment(Path(args.fuel_types))
    print(f"[backfill] enrichment tables: {sorted(enrichment)}")

    parent = _mass_column(steps["parent_pattern"], steps["library_id"], enrichment)
    child = _mass_column(steps["child_pattern"], steps["library_id"], enrichment)
    delta = child - parent

    steps["parent_fresh_enr_mass"] = parent
    steps["child_fresh_enr_mass"] = child
    steps["d_fresh_enr_mass"] = delta

    cols = [c for c in steps.columns if c not in NEW_COLUMNS]
    at = cols.index(_ANCHOR) + 1
    steps = steps[[*cols[:at], *NEW_COLUMNS, *cols[at:]]]

    finite = np.isfinite(delta)
    print(f"[backfill] d_fresh_enr_mass: {int(finite.sum())}/{len(delta)} finite, "
          f"nonzero {int((np.abs(delta) > 1e-12).sum())}, "
          f"max|d| {np.nanmax(np.abs(delta)):.6f}")
    conserving = steps["move_class"].isin(
        ["rewire_swap", "fresh_relocate", "batch_swap"]) & finite
    print(f"[backfill] conserving classes max|d| "
          f"{np.abs(delta[conserving.to_numpy()]).max():.3e}  "
          f"(rewire_swap / fresh_relocate / batch_swap conserve it exactly)")
    flip = (steps["move_class"] == "batch_flip").to_numpy() & finite
    if flip.any():
        print(f"[backfill] batch_flip           max|d| "
              f"{np.abs(delta[flip]).max():.6f}  n={int(flip.sum())}")

    if args.dry_run:
        print("[backfill] DRY RUN - steps.parquet not written")
        return 0
    backup = Path(str(args.steps) + ".bak_pre_fresh_enr_mass")
    if not backup.exists():
        backup.write_bytes(Path(args.steps).read_bytes())
        print(f"[backfill] backup -> {backup.name}")
    steps.to_parquet(args.steps, index=False)
    print(f"[backfill] wrote {args.steps}  ({len(steps.columns)} columns)")
    return 0


def fxy_columns(steps: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """The five new columns for ``steps``, as ``build_steps`` would emit them.

    ``f_xy`` is a MASTER reading (``FXYP`` in ``MAS_OUT``), so it can only be
    JOINED - by ``record_id``, the store's own identity, on BOTH ends of the
    edge.  ``improved_fxy`` reproduces ``build_steps``' rule exactly: pd.NA
    unless both ends converged AND both readings exist, else ``child < parent``.
    ``burn_state`` is recomputed with the shared implementation.
    """
    import mine_policy_corpus as M

    fxy = pd.to_numeric(
        store.set_index("record_id")["f_xy"], errors="coerce"
    ) if "f_xy" in store.columns else pd.Series(dtype=float)
    fxy = fxy[~fxy.index.duplicated(keep="last")]

    parent = steps["parent_record_id"].map(fxy).astype(float).to_numpy()
    child = steps["child_record_id"].map(fxy).astype(float).to_numpy()

    both = steps["both_converged"].fillna(False).astype(bool).to_numpy()
    known = both & np.isfinite(parent) & np.isfinite(child)
    improved = pd.array(child < parent, dtype="boolean")
    improved[~known] = pd.NA

    genomes: dict[str, object] = {}
    burn: list[str] = []
    for ppat, cpat in zip(steps["parent_pattern"], steps["child_pattern"],
                          strict=True):
        pg = genomes.get(ppat)
        if pg is None:
            pg = genomes[ppat] = M.genome_of(ppat)
        burn.append(M.move_burn_state(pg, ppat, cpat))

    return pd.DataFrame({
        "parent_f_xy": parent, "child_f_xy": child, "d_f_xy": child - parent,
        "improved_fxy": improved, "burn_state": burn,
    }, index=steps.index)


def cmd_backfill_fxy(args) -> int:
    import mine_policy_corpus as M

    steps_path = Path(args.steps)
    steps = pd.read_parquet(steps_path)
    print(f"[fxy] {len(steps)} rows, {len(steps.columns)} columns")
    already = [c for c in M.FXY_SCHEMA_COLUMNS if c in steps.columns]
    if already and not args.force:
        print(f"[fxy] already present: {already} - nothing to do")
        return 0

    import pyarrow.parquet as pq

    store_path = Path(args.store)
    if "f_xy" not in pq.ParquetFile(store_path).schema_arrow.names:
        print(f"[fxy] {store_path} has no f_xy column - nothing to join; "
              "back-fill the store first (lpopt/tools/backfill_fxy.py)")
        return 1
    store = pd.read_parquet(store_path, columns=["record_id", "f_xy"])
    print(f"[fxy] store: {len(store)} records, "
          f"{int(pd.to_numeric(store['f_xy'], errors='coerce').notna().sum())} "
          "with an F_xy label")

    new = fxy_columns(steps, store)
    have_p = np.isfinite(new["parent_f_xy"].to_numpy())
    have_c = np.isfinite(new["child_f_xy"].to_numpy())
    both = have_p & have_c
    print(f"[fxy] join coverage: parent {int(have_p.sum())}/{len(steps)}  "
          f"child {int(have_c.sum())}/{len(steps)}  "
          f"BOTH ENDS {int(both.sum())}/{len(steps)} "
          f"({both.mean():.1%})")
    print(f"[fxy] improved_fxy known on {int(new['improved_fxy'].notna().sum())} "
          f"rows, True on {int((new['improved_fxy'] == True).sum())}")  # noqa: E712
    print("[fxy] burn_state mix: "
          f"{new['burn_state'].value_counts().to_dict()}")
    by_src = (pd.DataFrame({"lineage_source": steps["lineage_source"],
                            "both": both})
              .groupby("lineage_source")["both"].agg(["sum", "size"]))
    print("[fxy] rows with F_xy on both ends, by lineage_source:")
    print(by_src.to_string())

    for col in M.FXY_SCHEMA_COLUMNS:
        steps[col] = new[col]

    if not args.apply:
        print("[fxy] DRY RUN (default) - steps.parquet not written; "
              "re-run with --apply")
        return 0
    backup = Path(args.backup)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(steps_path.read_bytes())
        print(f"[fxy] backup -> {backup}")
    else:
        print(f"[fxy] backup already exists, kept: {backup}")
    steps.to_parquet(steps_path, index=False)
    print(f"[fxy] wrote {steps_path}  ({len(steps.columns)} columns)")
    return 0


def cmd_verify(args) -> int:
    """Re-mine a campaign already in the corpus; every column must match."""
    import ablation_analyze as A

    A.STORE = BASE / "data/store/records.parquet"
    A.STEPS = Path(args.steps)
    fresh = A.build_wave_steps(args.campaign, None if args.lineage == "native"
                               else args.lineage)
    have = pd.read_parquet(args.steps)
    have = have[have["campaign"] == args.campaign]
    print(f"[verify] {args.campaign}: canonical {len(have)} rows, "
          f"re-mined {len(fresh)} rows")
    if len(have) != len(fresh):
        print("[verify] ROW COUNT MISMATCH")
        return 1
    key = ["parent_record_id", "child_record_id"]
    a = have.sort_values(key).reset_index(drop=True)
    b = fresh[have.columns].sort_values(key).reset_index(drop=True)
    bad = []
    for col in have.columns:
        x, y = a[col], b[col]
        # ``None`` (parquet round-trip) and ``pd.NA`` (concat of an all-null
        # object column) are the same absence; compare missingness separately so
        # a representation difference is not reported as data drift.
        if not bool((x.isna() == y.isna()).all()):
            bad.append(col)
            continue
        keep = ~x.isna().to_numpy()
        if not keep.any():
            continue
        x, y = x[keep], y[keep]
        if x.dtype.kind == "f":
            ok = np.allclose(x.astype(float), y.astype(float),
                             rtol=0, atol=1e-12, equal_nan=True)
        else:
            ok = bool((x.astype(str) == y.astype(str)).all())
        if not ok:
            bad.append(col)
    print(f"[verify] mismatching columns: {bad or 'NONE - byte-identical'}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default=str(STEPS))
    ap.add_argument("--fuel-types", default=str(FUEL_TYPES))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("backfill")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("backfill-fxy")
    p.add_argument("--store", default=str(STORE))
    p.add_argument("--backup", default=str(FXY_BACKUP))
    p.add_argument("--apply", action="store_true",
                   help="actually write (default is a dry run)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_backfill_fxy)

    p = sub.add_parser("verify")
    p.add_argument("--campaign", default="fpcamp_minfr_T6T4")
    p.add_argument("--lineage", default="native")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
