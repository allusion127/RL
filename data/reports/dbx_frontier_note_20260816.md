# Composition-level frontier from the feasible database — screening prior

**Date** 2026-08-16 · **Table** `data/reports/dbx_frontier_table.csv` (286 rows × 74 cols)
**Source (read-only)** `data/reports/scoping_mesh_20260815/feasible_database.xlsx` sheet `cores`
· **Store coverage joined from** `data/store/records.parquet` (`library_id = ga80`, read-only)

Companion to `dbx_lrm_fit_20260816.md`. Evidence for
`data/reports/reloadmap_methodology_20260816.md` (written elsewhere; not edited here).

> **E14 축 각주 (2026-08-20 확정)**: 이 노트의 **`pinmax_node_GWd` 열이 우리
> 프로그램의 규제축과 같은 계층**이다 — 공식 관측량은 **핀 axial peak**, 한도는
> **80 GWd/tU**(`data/reports/pinbu_definition_20260820.md`). 본문의 **75 GWd/tU 눈금은 설계 스크리닝**이지
> 인허가 한도가 아니며, `rodavg` 열은 **보조 관측량** 쪽이다. 두 열을 섞어 읽지 말
> 것. 다만 DB 수치와 우리 실측 사이에는 공유 셀 기준 +9~20% 의 미분리 수준차가
> 있다(`pinbu_audit_20260820.md` §1.5) — DB node 값을 우리 값의 대리로 쓰지 않는다.

---

## 1. What the table is

One row per **(pair, feed, composition-split bucket)** — the finest cell at which the
database still has a population to take a minimum over. 286 non-empty cells across
20 pairs, 6 feeds and 6 split buckets, covering all 6,113 cores.

*Split* = `n_type1 / (n_type1 + n_type2)` = the fraction of the feed batch that is the
light-load (`role: L`) type. Observed range 0.504–0.832; bucketed
`[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.90)` → labels `50-55 … 75-90`
(67 / 76 / 62 / 41 / 24 / 16 cells).

Each row carries, for the cell's **minimum-F_r core** (the frontier point): `cid`, exact
`n_type1 / n_type2`, EFPD, `B_c`, mass-balance `B_d`, reported discharge burnup, CBC_max and
margin, F_q, AO band, both pin-burnup metrics and all six `ge62/68/75` flags, the campaign
values, and `metrics_source`. Plus cell aggregates (F_r p10/median, EFPD mean/max, B_d mean,
pin-burnup min/mean, the six exceedance fractions, `n_campaign_verified`), the cell's
**longest-cycle** core as the second Pareto corner, and the store-coverage join
(`n_store_rows`, `n_store_valid`, `store_f_r_min`, `store_cyclen_max`, `store_gap_f_r`).

**Pin-burnup honesty.** The 986 `metrics_source = campaign` rows carry no pin-burnup at all
(`pinmax_rodavg_GWd` NaN, and all six `*_ge*` booleans sit at `False` rather than null).
`n_pinmax_known` counts the rows where it is real, `best_pinmax_known` says whether the
frontier core itself has it, and `frac_node_ge75` / `frac_rodavg_ge75` are computed **only**
over rows with a known value (NaN if the cell has none). 282 of 286 cells have at least one;
4 are campaign-only.

## 2. The shape of the frontier

Best DB-proven F_r by feed:

```
feed      101      105      109      113      117      121
F_r min  1.4958   1.5024   1.5290   1.5226   1.5129   1.4971
```

The flatness floor is **U-shaped in feed** — it bottoms at both ends (f101 1.4958,
f121 1.4971) and peaks in the middle (f109 1.5290). Feed 109 is simultaneously the
longest-cycle-per-enrichment column (the "f109 ridge" of the LRM report, §4) and the
*worst* flatness column. That is a real, exploitable trade and it is invisible on the
model mesh, which does not carry f101 at all.

Split bucket matters much less than feed: the median cell-minimum F_r is 1.529–1.535 in
every bucket. The global best cores sit at 60-65 (1.4958) and 65-70 (1.4971), but the
spread across buckets is 0.006 — **do not screen on the split; screen on (pair, feed)
and let the inner loop pick the split.**

The binding constraint flips with feed:

* **f101–f105** buy flatness (F_r 1.496–1.51) and long discharge burnup (B_d 54–57
  GWd/tU) but **every** such core in the database exceeds 75 GWd/tU node pin burnup
  (`frac_node_ge75 = 1.0`, rod-average 78–81 GWd/tU).
* **f117–f121** stay clear of the pin-burnup cliff (node 69–73 GWd/tU, `frac_node_ge75 = 0`)
  at a flatness cost of ~0.02–0.03 in F_r and 1.5–3 GWd/tU of discharge burnup.

For an LEU+ licensing campaign the second family is the defensible one, and it is also the
family the store has barely touched.

## 3. Store coverage — what we have never run

`n_store_valid` is the count of *valid* `ga80` records at the same normalised
(pair, feed). 83 of 286 frontier cells have **zero**. By pair:

| pair | DB cores | DB best F_r | store valid rows (all feeds) |
|---|---:|---:|---:|
| `E5_E6` | 56 | 1.5273 | **0** — type not in `fuel_types.parquet` |
| `J7_J8` | 416 | 1.5124 | **0** — type not in `fuel_types.parquet` |
| `A8_A2` | 2 | 1.5427 | 4 |
| `J3_J4` | 157 | 1.5171 | 11 |
| `N5_N6` | 31 | 1.5362 | 18 |
| `K5_K6` | 478 | 1.5217 | 19 |
| `K3_K4` | 15 | 1.5249 | 50 |
| `H1_H2` | 6 | 1.5273 | 157 |
| … 12 pairs with 183–2073 … | | | |

Where both sides have data, the gap is brutal: over the 30 screened cells with any store
coverage, `store_f_r_min − F_r_min` has **median +0.330** (IQR 0.257–0.360, max 0.381).
Our best recorded F_r at `N3_N4 / f105 / 60-65` is 1.891; the database proves 1.510 there.
Part of that is objective mismatch (most store rows were not produced by an F_r-minimising
search), but a 0.33 gap is not an objective artefact — it is unexplored space.

## 4. The ten most attractive unexplored-by-us cells

**Screen** (deliberately simple, so it can be re-run with different weights straight off
the CSV): `n_cores ≥ 5` **and** `F_r_min ≤ 1.535` **and** `n_store_valid ≤ 25`.
71 cells pass. **Rank** by `F_r_min` ascending, then `best_EFPD` descending.
**Tier A** additionally requires the frontier core to clear the 75 GWd/tU node pin-burnup
cliff with a *known* value. All ten below are Tier A and all are `equilibrium_ncyc12`
(not campaign extrapolations).

| # | pair | feed | split | feed split (n1×n2) | n cores | **DB F_r** | EFPD | B_d | CBC_max (margin) | F_q | pin rod-avg / node | store valid | store best F_r |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `L3_L4` | 121 | 50-55 | 65 × 56 | 23 | **1.5217** | 668.0 | 51.94 | 1347 (173) | 1.883 | 66.3 / 72.3 | 24 | 1.522 |
| 2 | `K5_K6` | 117 | 55-60 | 68 × 49 | 87 | **1.5229** | 648.3 | 52.13 | 1342 (138) | 1.893 | 64.1 / 70.1 | **0** | — |
| 3 | `L3_L4` | 121 | 60-65 | 73 × 48 | 13 | 1.5258 | 666.6 | 51.83 | 1329 (191) | 1.915 | 66.7 / 73.0 | 24 | 1.522 |
| 4 | `L3_L4` | 121 | 55-60 | 72 × 49 | 18 | 1.5292 | 670.4 | 52.13 | 1417 (103) | 1.915 | 66.5 / 72.5 | 24 | 1.522 |
| 5 | `J7_J8` | 121 | 50-55 | 64 × 57 | 12 | 1.5300 | 642.1 | 49.92 | 1268 (172) | 1.908 | 63.6 / 69.3 | **0** ★ | — |
| 6 | `J7_J8` | 121 | 60-65 | 73 × 48 | 13 | 1.5307 | 644.0 | 50.07 | 1326 (114) | 1.909 | 63.8 / 69.5 | **0** ★ | — |
| 7 | `J7_J8` | 121 | 55-60 | 68 × 53 | 17 | 1.5314 | 644.6 | 50.12 | 1323 (117) | 1.919 | 63.8 / 69.5 | **0** ★ | — |
| 8 | `K5_K6` | 117 | 60-65 | 76 × 41 | 60 | 1.5320 | 649.3 | 52.21 | 1372 (108) | 1.900 | 64.5 / 70.8 | **0** | — |
| 9 | `K5_K6` | 121 | 50-55 | 65 × 56 | 12 | 1.5339 | 655.5 | 50.97 | 1302 (178) | 1.898 | 65.3 / 71.3 | 19 | 1.533 |
| 10 | `J3_J4` | 121 | 50-55 | 65 × 56 | 7 | 1.5340 | 639.4 | 49.71 | 1232 (208) | 1.912 | 63.4 / 69.1 | 11 | **1.501** † |

★ = pair uses `J7`/`J8`, which **do not exist in `data/store/fuel_types.parquet`** — these
cells are unreachable by our pipeline today, not merely unexplored. Merging
`config/fuel_types_dbx_extracted.yaml` is the prerequisite for running rows 5–7 at all.

† = the store's 11 rows at `J3_J4 / f121` already reach F_r 1.501, **flatter than the
database's 1.534 in this bucket and than the pair's DB-wide best of 1.5171**. Row 10 clears
the screen only because the DB population there is thin (7 cores); it is not a prize — it
is a cell where we are already ahead of the database. Same caution, milder, for rows 1/3/4
(`store_f_r_min` 1.522, i.e. parity).

**Top-3 campaign candidates** (reading the table with the LEU+ constraint set in mind):

1. **`K5_K6` @ f117, split 0.55–0.60.** F_r 1.5229 with 648 EFPD and node pin burnup
   70.1 GWd/tU — the best flatness-per-pin-burnup in the whole screened set. The store has
   **19 valid `K5_K6` rows total across all feeds** against 478 database cores, and *zero*
   at f117. 87 database cores in this single cell, so the frontier value is well supported.
   Adjacent cell #8 (60-65, 60 cores) gives a second, independent confirmation.
2. **`L3_L4` @ f121, split 0.50–0.55.** The flattest Tier-A cell (1.5217) and the longest
   cycle in the top ten (668–670 EFPD, B_d 51.9). Store coverage is 24 rows at this feed
   and its recorded best F_r (1.522) already *matches* the database — so this is the one
   cell where our pipeline is demonstrably on the frontier. Cheap to verify, high value as
   a **calibration anchor** rather than a discovery.
3. **`J7_J8` @ f121 (all three splits).** 416 database cores on a pair we cannot featurize
   at all. F_r 1.530–1.531, 642–645 EFPD, and among the lowest pin burnups in the screen
   (rod-avg 63.6, node 69.3 GWd/tU — only `J3_J4 / f121` is marginally lower at 63.4 / 69.1)
   with 172 ppm of boron margin. It is not the flattest
   cell, but it is the largest *capability* gap: three cells, zero store rows, and a fuel
   type the store has never seen. The LRM backbone also mis-predicts `J7_J8` by −2.0 EFPD
   and `E5_E6` by +5.6 EFPD (worst two pairs), so a campaign here buys model information as
   well as design information.

## 5. Cells to explicitly *not* chase

`N3_N4 @ f101 / 55-65` and `N1_N2 @ f105 / 55-65` carry the database's flattest cores
(F_r 1.496–1.510) and near-zero store coverage, so a naive "lowest F_r, least explored"
screen puts them first. They are excluded from Tier A because **100 % of their cores exceed
75 GWd/tU node pin burnup** (rod-average 78.6–80.5, node 85.6–87.7 GWd/tU). Unless the
pin-burnup ceiling is relaxed, those cells are a licensing dead end no matter how flat they
are. They are still in the CSV — filter on `frac_node_ge75` to bring them back.

Four cells (`n_pinmax_known = 0`) are campaign-only and have **no** pin-burnup evidence
either way; treat their `*_ge75 = False` flags as unknown, not as clearance.
