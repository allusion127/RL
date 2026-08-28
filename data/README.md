# `data/` — local unified store and campaign artifacts

This directory holds the **local original** of all lpopt data (the remote GPU
server keeps copies only; see plan section 4.7). It is intentionally kept out of
version control except for this README (acts as the `.gitkeep`).

Expected layout once the pipeline runs (milestones M2+):

```
data/
├─ store/
│  ├─ records.parquet      # one row per unique LP evaluation (record_id = 64-hex)
│  ├─ maps.npz             # EDIT5 per-cell labels (multitask spatial regularizer)
│  └─ fuel_types.parquet   # physics fuel features per (type, library_id)
├─ produce/
│  └─ ledger.jsonl         # append-only production-campaign ledger
├─ splits/                 # S0/S1/S2/S3/S4 holdout definitions
└─ reports/                # extract_report.md, coverage heatmaps, ...
```

Nothing here yet at milestone M0.
