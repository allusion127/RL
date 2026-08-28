# Dataset A extraction report

- generated: 2026-07-16 20:00:14
- wall time: 68.6 s
- store: `C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL\data\store`
- unique records: **38854**  (audit ground truth: 38,854)
- converged: 38851   non-converged: 3
- fresh-footprint uniqueness (ESS proxy): **14682** (audit expects ~14,682)
- unresolved libraries: 0   genome parse failures: 0   e_core missing: 0

## Per-file line / record / unique counts

| file | status | lines | records | new unique |
|---|---|---:|---:|---:|
| 0_Case/sa_2b_cache.jsonl | OK | 14710 | 14709 | 14709 |
| 0_Case/sa_2b_cache.stale-12838589.jsonl | OK | 2419 | 2418 | 2418 |
| 0_Case/sa_2b_cache.stale-2c04d78c.jsonl | OK | 3343 | 3342 | 3342 |
| 0_Case/sa_2b_cache.stale-2df4a80d.jsonl | OK | 3 | 2 | 2 |
| 0_Case/sa_2b_cache.stale-40a80759.jsonl | OK | 1253 | 1252 | 1252 |
| 0_Case/sa_2b_cache.stale-925b1136.jsonl | OK | 10949 | 10948 | 10948 |
| 0_Case/sa_2b_cache.stale-b01338df.jsonl | OK | 1231 | 1230 | 1230 |
| 0_Case/sa_2b_cache.stale-fb857c7a.jsonl | OK | 4320 | 4319 | 4319 |
| eqlp_ws/sa_2b_cache.jsonl | OK | 286 | 285 | 285 |
| eqlp_ws/sa_2b_cache.stale-ee2b1ddf.jsonl | OK | 359 | 358 | 349 |
| eqlp_ws_rev02/sa_2b_cache.jsonl | OK | 10 | 9 | 0 |
| **total** | | 38883 | 38872 | 38854 |

## Per-library record counts

| library_id | records | cbc_max harvested | cbc coverage |
|---|---:|---:|---:|
| 260624 | 29976 | 25657 | 85.6% |
| 5.8_5.1 | 8244 | 0 | 0.0% |
| legacy_a | 634 | 634 | 100.0% |

resolution source totals: run_meta=35802, name_pattern=3052
(5.8_5.1 / older-era case dirs were purged after caching -> boc_only; 260624 and legacy_a case dirs survive.)

## Top-30 case pairs

| pair | records |
|---|---:|
| C1_C2 | 5728 |
| C1_C4 | 4935 |
| C3_C6 | 4390 |
| A01_B05 | 3239 |
| C1_C6 | 3055 |
| A01_A02 | 2021 |
| C01_C04 | 1589 |
| C01_C02 | 1517 |
| A01_A04 | 1278 |
| B1_C6 | 1049 |
| B01_B03 | 850 |
| B3_C6 | 818 |
| C5_C6 | 777 |
| B1_C2 | 714 |
| A0_A1 | 634 |
| C2_C3 | 590 |
| B5_C6 | 341 |
| B01_B05 | 321 |
| B1_C4 | 297 |
| C01_C06 | 248 |
| C03_C06 | 222 |
| C2_C5 | 150 |
| C05_C06 | 135 |
| B1_B4 | 131 |
| B2_C1 | 129 |
| B3_B4 | 116 |
| B4_C1 | 114 |
| B3_C2 | 111 |
| B6_C1 | 109 |
| C4_C5 | 103 |

distinct pairs: 139

## CBC recompute coverage

- cbc_kind="max" (harvested): 26291 / 38854 = **67.7%**
- cbc_kind="boc_only" (residual): 12563

## (feed x e_core-bin) 2-D support histogram

| feed \ e_core | 5.3-5.4 | 5.4-5.5 | 5.5-5.6 | 5.6-5.7 |
|---|---:|---:|---:|---:|
| 121 | 14747 | 18803 | 4670 | 634 |

## Harvest

- case dirs indexed: 37476
| status | count |
|---|---:|
| inp_err | 2 |
| no_sum | 3 |
| nomatch | 11159 |
| ok | 26312 |

harvest IO / parse failures: 2   (missing MAS_SUM -> boc_only: 3)

<!-- lpopt:dataset-b -->

# Dataset B extraction report

- generated: 2026-07-16 21:38:07
- wall time: 1.0 s
- store: `C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL\data\store`
- event-log entries (raw): 3576   unique (case,digest): 3449
- records written: **574**  (event 72 + manifest 502)
- converged: 574   non-converged: 0   error rows: 0
- pattern recovery: 72 / 3449 = **2.1%** (unrecovered dropped: 3377)
- parent lineage: 65 resolved, 7 unresolved (parent not ingested)
- feed / pair mismatches (recovered vs case string): 0 / 0
- genome parse failures: 0   e_core missing: 0

## Per-run event-log counts (raw, pre-dedup)

| campaign (runs_flow ts) | case | entries | feasible | errors | eq_ok |
|---|---|---:|---:|---:|---:|
| 20260712_193742 | K1_K2/feed-121 | 24 | 0 | 4 | 19 |
| 20260712_203925 | K1_K2/feed-121 | 240 | 0 | 14 | 225 |
| 20260712_233433 | K1_K2/feed-121 | 600 | 0 | 9 | 590 |
| 20260713_061541 | K1_K2/feed-121 | 600 | 70 | 17 | 582 |
| 20260713_140758 | K3_K4/feed-121 | 592 | 0 | 10 | 581 |
| 20260713_194353 | K3_K4/feed-121 | 304 | 0 | 8 | 295 |
| 20260713_223509 | K5_K6/feed-121 | 344 | 0 | 9 | 334 |
| 20260714_024213 | K3_K4/feed-121 | 600 | 0 | 17 | 582 |
| 20260714_084703 | K5_K6/feed-121 | 248 | 2 | 10 | 237 |
| 20260714_114757 | K5_K6/feed-121 | 24 | 0 | 5 | 18 |

**Audit run 20260713_061541 / K1_K2**: 600 labels / 70 feasible / 17 errors (ground truth 600 / 70 / 17 -> MATCH)

## Per-pair record counts

| case_pair | records |
|---|---:|
| K1_K2 | 153 |
| J1_J2 | 91 |
| E1_E2 | 44 |
| G3_G4 | 35 |
| N1_N2 | 32 |
| L1_L2 | 31 |
| L3_L4 | 24 |
| H3_H4 | 23 |
| J5_J6 | 22 |
| N3_N4 | 21 |
| E3_E4 | 18 |
| N5_N6 | 17 |
| K3_K4 | 15 |
| K5_K6 | 14 |
| L5_L6 | 14 |
| J3_J4 | 10 |
| H1_H2 | 6 |
| A8_A2 | 4 |

## Pattern recovery

- deck files scanned: 1173
- read errors (dehydrated placeholders): 97
- parse errors: 0
- unique recovered digests: 694
- dehydrated manifest-core roots skipped: ['ga_campaign_K1_K2', 'ga_campaign_K5_K6', 'ga_rl_package']

## Manifests (dehydrated-tolerant)

| root | status | rows | joined | row errors | detail |
|---|---|---:|---:|---:|---|
| FEASIBLE_PACKAGE | OK | 502 | 502 | 0 |  |
| ga_campaign_K1_K2 | DEHYDRATED | 0 | 0 | 0 | manifest unreadable: [Errno 22] Invalid argument |
| ga_campaign_K5_K6 | DEHYDRATED | 0 | 0 | 0 | manifest unreadable: [Errno 22] Invalid argument |
| ga_rl_package | DEHYDRATED | 0 | 0 | 0 | manifest unreadable: [Errno 22] Invalid argument |

## (feed x e_core-bin) 2-D support histogram

| feed \ e_core | 5.0-5.1 | 5.1-5.2 | 5.2-5.3 | 5.3-5.4 | 5.4-5.5 | 5.5-5.6 |
|---|---:|---:|---:|---:|---:|---:|
| 117 | 12 | 65 | 58 | 0 | 0 | 0 |
| 121 | 54 | 58 | 124 | 69 | 70 | 64 |
