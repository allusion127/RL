# `data/` — local unified store and campaign artifacts

This directory holds the **local original** of all lpopt data (the remote GPU
server keeps copies only; see plan section 4.7). It is intentionally kept out of
version control except for this README (which acts as the `.gitkeep`), the
`reports/`, `reference/` and `design/` manifests, and the champion metadata in
`models/s1j/` — see the repository `.gitignore` for the exact allow-list.

## Layout

```
data/
├─ store/                    # THE unified store — every other artifact is derived
│  ├─ records.parquet        # one row per unique LP evaluation (record_id = 64-hex)
│  ├─ maps.npz               # harvested EDIT5/EDIT6 arrays, keyed by record_id
│  ├─ fuel_types.parquet     # physics fuel features per (type, library_id)
│  ├─ flat_scale.json        # per-cell flat_power normalizers (fit_flat_scale)
│  └─ map_calibration.json   # per-cell map-head bias / sigma (fit_map_calibration)
├─ maps/                     # (when present) map exports split out of maps.npz
├─ splits/                   # S0…S4 frozen holdout definitions (JSON)
├─ policy/                   # mined policy corpora: steps / elites / sa_lineage
├─ produce/                  # production-campaign ledger.jsonl + run logs
├─ models/                   # trained checkpoints, one dir per training run
├─ design/                   # MASTER package, decks, fuel-type registry, synth decks
├─ campaigns/                # per-campaign run trees (labels.jsonl, waves/, report.md)
├─ curriculum/               # curriculum-agent state + per-cell logs
├─ reference/                # reference decks / assets
└─ reports/                  # design docs, results readouts, coverage heatmaps
```

`*.bak_*` siblings are point-in-time snapshots taken before a destructive
operation; they are never read by the code.

## `records.parquet` columns

The schema is **append-only** (`lpopt/data/schema.py` asserts it at import):

* **36 FROZEN columns** — the prefix every historical parquet already has. Their
  names, order and types never change.
* **LATE columns** — appended after the freeze, and read back as `NaN` from any
  parquet written before they existed. `StoreReader.ensure_schema_columns` fills
  them in, so an old store and a new one merge without migration:

  | column | source | meaning |
  |---|---|---|
  | `node_peak` | EDIT5 BOC map | max of the 69 assembly power slots — **BOC assembly radial peaking, NOT F_xy** |
  | `map_cov` | EDIT5 BOC map | coefficient of variation of the same map |
  | `max_rod_avg_burnup` | `MAS_PPI` | rod-average pin burnup peak (secondary observable) |
  | `f_xy` | `MAS_OUT` `FXYP` | **pin PLANAR peaking** — max over every depletion step of the final equilibrium cycle. Licensing limit **1.65** (user decision 2026-08-29); the objective of `objective = "min_fxy"` |
  | `f_xya` | `MAS_OUT` `FXYA` | assembly planar peaking (report-only sibling of `f_xy`) |

  `f_xy` / `f_xya` are absent from `MAS_SUM`, so they exist only for rows whose
  final-cycle `MAS_OUT` survived — i.e. runs with `[verify] harvest_maps = true`
  (which forces `keep_success`). Everything else carries `NaN`, and that is
  **UNKNOWN, not compliant**: the search predicate
  (`campaign.is_feasible_search`) lets a missing `f_xy` pass so a campaign is not
  starved before the first label, while the delivery predicate
  (`campaign.is_deliverable`) refuses it. Retro-labelling of existing rows is
  `python -m lpopt.tools.backfill_fxy`.
