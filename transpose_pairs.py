"""Transpose-pair experiment: measure MASTER label reproducibility DIRECTLY.

A quarter-core pattern and its diagonal transpose (qi,qj)->(qj,qi) describe the
SAME physical reactor in a different representation: reflecting the quarter about
the diagonal reflects the whole assembled core, which is a symmetry of the
machine.  But they are DIFFERENT records to us — different pattern digest, hence
different ``record_id`` and different rot61 cache key — so the harness really runs
MASTER twice and the two label sets can be compared.

Any difference between the pair's labels is pure irreducible noise: solver
convergence, mesh/ordering effects, restart bookkeeping.  That converts every
"label-precision ceiling" number from an ESTIMATE (quantization bound) into a
MEASUREMENT.

Usage
  python transpose_pairs.py --deck fill_199.inp --n 24 --dry-run   # plan only
  python transpose_pairs.py --deck fill_199.inp --n 24             # runs MASTER
Report
  python transpose_pairs.py --report runs/transpose_pairs/outcomes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_LABELS = ("f_r", "f_q", "cbc_max", "cyclen", "ao_abs",
           "max_assembly_burnup", "max_pin_burnup")


def transpose_pattern(pat):
    """Diagonal transpose of a quarter-core pattern (same physics, new encoding).

    A plain slot swap (qi,qj)->(qj,qi) produces an INVALID mirror-69 pattern and
    MASTER refuses to stage it (forensic 20260725: 24 wasted calls, every
    transpose side failed).  Two things must move with the coordinates:

    * the BURNED card's source coordinate, which is itself a quarter cell, and
    * the ROTATION, because the transpose swaps the horizontal and vertical arms
      and their conventions differ -- ``validate_quarter_conventions`` requires
      rotation 1 on ``vertical_axis`` slots and 2 elsewhere (fresh cards are
      always rotation 0).

    The result validates, keeps the same feed, and is an involution
    (``T(T(p)) == p``) -- it is a reflection, so applying it twice is identity.
    """
    import dataclasses
    from lpopt.data.geometry import (QuarterCell, cell_of_label, cell_of_slot,
                                     label_of_cell, slot_index_of)
    from lpopt.vendor.masterrl.domain import SLOTS, Pattern

    items = list(pat.items)
    out: list = [None] * len(items)
    for s, item in enumerate(items):
        c = cell_of_slot(s)
        dest = slot_index_of(QuarterCell(c.qj, c.qi))
        if item.is_fresh:
            out[dest] = dataclasses.replace(item, rotation=0)
        else:
            src = cell_of_label(item.x, item.y)
            nx, ny = label_of_cell(QuarterCell(src.qj, src.qi))
            rot = 1 if SLOTS[dest].orbit_class == "vertical_axis" else 2
            out[dest] = dataclasses.replace(item, x=nx, y=str(ny), rotation=rot)
    if any(x is None for x in out):
        raise ValueError("transpose left holes - geometry mismatch")
    tp = Pattern(tuple(out))
    tp.validate_quarter_conventions()      # fail LOUDLY, never ship a bad card
    return tp


def fom_labels(fom):
    """Map a vendor ``FOM`` onto the store's label names.

    The FOM does NOT expose ``ao_abs`` / ``max_assembly_burnup`` -- it has
    ``ao_min``/``ao_max`` and ``max_burnup`` (``max_burnup_assembly`` is the
    assembly ID, not a value).  Reading the store names off a FOM silently
    yields None for those columns, which is how the first run produced
    all-null comparisons.
    """
    def g(n):
        v = getattr(fom, n, None)
        return float(v) if v is not None else None

    ao_min, ao_max = g("ao_min"), g("ao_max")
    ao_abs = None
    if ao_min is not None or ao_max is not None:
        ao_abs = max(abs(ao_min or 0.0), abs(ao_max or 0.0))
    return {"f_r": g("f_r"), "f_q": g("f_q"), "cbc_max": g("cbc_max"),
            "cyclen": g("cyclen"), "ao_abs": ao_abs,
            "max_assembly_burnup": g("max_burnup"),
            "max_pin_burnup": g("max_pin_burnup")}


def pick_pairs(store_dir: str, library_id: str, n: int, seed: int = 0):
    """``n`` (original, transpose) pairs drawn from converged store rows."""
    from lpopt.data.schema import unpack_pattern

    df = pd.read_parquet(Path(store_dir) / "records.parquet")
    df = df[(df["converged"]) & (df["library_id"] == library_id)]
    if not len(df):
        raise SystemExit(f"no converged {library_id} rows in {store_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    pairs = []
    for i in idx:
        row = df.iloc[int(i)]
        pat = unpack_pattern(row["pattern"])
        tp = transpose_pattern(pat)
        if tp.digest == pat.digest:
            continue                      # self-symmetric: measures nothing
        pairs.append({
            "case_pair": row["case_pair"], "feed": int(row["feed"]),
            "orig_digest": pat.digest, "tp_digest": tp.digest,
            "orig": pat, "tp": tp,
            "orig_labels": {k: (float(row[k]) if pd.notna(row.get(k)) else None)
                            for k in _LABELS if k in row},
        })
    return pairs


def report(path: str) -> None:
    """Summarize |orig - transpose| per label = the measured noise floor."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [d for d in data if d.get("tp_labels") and d.get("orig_labels")]
    print(f"pairs with both sides converged: {len(rows)}")
    print(f"{'label':22s} {'n':>4s} {'mean|d|':>10s} {'p95|d|':>10s} {'max|d|':>10s}")
    for k in _LABELS:
        d = [abs(r["orig_labels"][k] - r["tp_labels"][k]) for r in rows
             if r["orig_labels"].get(k) is not None
             and r["tp_labels"].get(k) is not None]
        if d:
            print(f"{k:22s} {len(d):4d} {np.mean(d):10.5f} "
                  f"{np.percentile(d,95):10.5f} {np.max(d):10.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-root", default="runs/transpose_pairs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()

    if a.report:
        report(a.report)
        return
    if not a.deck:
        raise SystemExit("--deck is required (or use --report)")

    from lpopt.config import load_config
    cfg = load_config(a.deck)
    lib = cfg.model.library_id
    pairs = pick_pairs(cfg.model.store_dir, lib, a.n, a.seed)
    print(f"deck={a.deck} library={lib} pairs={len(pairs)} "
          f"(= {2*len(pairs)} MASTER calls)")
    for p in pairs[:5]:
        print(f"  {p['case_pair']:>32s} f{p['feed']}  {p['orig_digest']} -> {p['tp_digest']}")
    if a.dry_run:
        print("DRY-RUN: no MASTER. Re-run without --dry-run (needs LPOPT_WORKER=1).")
        return

    import os
    if os.environ.get("LPOPT_WORKER") != "1":
        raise SystemExit("refusing to run MASTER without LPOPT_WORKER=1")

    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.search.resolver import build_case_resolver
    from lpopt.search.verify import WaveEntry, WaveVerifier
    from lpopt.vendor.masterrl.dataset import CaseKey

    root = Path(a.deck).resolve().parent
    fuel = FuelLibrary.from_parquet(root / "data/store/fuel_types.parquet")
    res = build_case_resolver(cfg, fuel, lib)
    run_dir = root / a.run_root
    run_dir.mkdir(parents=True, exist_ok=True)

    entries: list[WaveEntry] = []
    tags: list[tuple[int, str]] = []
    for i, p in enumerate(pairs):
        ck = CaseKey(pair=p["case_pair"], feed=p["feed"])
        assets = res.resolve(ck)
        for side in ("orig", "tp"):
            entries.append(WaveEntry(pattern=p[side], case_key=ck,
                                     resolved_assets=assets,
                                     meta={"transpose_side": side, "pair_ix": i}))
            tags.append((i, side))

    v = WaveVerifier(
        run_dir=run_dir / "master",
        package_root=root / cfg.verify.package_root,
        executable=cfg.master.executable,
        workers=cfg.produce.workers, use_all_cores=cfg.produce.use_all_cores,
        host_reserve=cfg.produce.host_reserve, timeout=cfg.master.timeout,
        max_cycles=cfg.master.max_cycles, consecutive=cfg.master.consecutive,
        resolver=res, harvest_maps=bool(cfg.verify.harvest_maps),
    )
    outcomes = v.evaluate_wave(entries)

    for (i, side), oc in zip(tags, outcomes, strict=True):
        if oc.status != "converged" or oc.fom is None:
            continue
        pairs[i][f"{side}_labels_run"] = fom_labels(oc.fom)
    out = []
    for p in pairs:
        out.append({
            "case_pair": p["case_pair"], "feed": p["feed"],
            "orig_digest": p["orig_digest"], "tp_digest": p["tp_digest"],
            "orig_labels": p.get("orig_labels_run"),
            "tp_labels": p.get("tp_labels_run"),
        })
    dest = run_dir / "outcomes.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    report(str(dest))


if __name__ == "__main__":
    main()
