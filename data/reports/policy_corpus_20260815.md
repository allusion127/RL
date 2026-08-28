# Policy corpus mining — improvement-step corpus (stage 1)

Generated 2026-08-15 by `mine_policy_corpus.py` (read-only over `data/store/records.parquet`).

Goal: supervised data for a **learned move-proposal policy** — not a surrogate. One row of `data/policy/steps.parquet` is one board->board *move* the campaigns actually made, with the move class inferred and the outcome labeled.

## 1. Headline

- store rows: **72,685** (64,405 converged)
- step pairs emitted: **27,458** — **21,766** recovered from the Dataset A MOCHA SA logs (section 6a) plus **5,692** from the store's own `parent_record_id` lineage. Of the total:
  - **same-cell MOVES: 19,820** — parent and child share (pair, feed, library). *This is the policy corpus.*
  - cross-cell TRANSFERS: **7,638** — feed-morph / pair-transfer re-seedings (`elite_perturb`, `transfer`, `g3_elite_boundary`, and MOCHA's fuel-family repaints). Kept in the parquet under `cross_cell=True` because they are real physics, but they are **not moves in a fixed action space** and are excluded from every policy-facing table below. They are 28% of the raw lineage — down from 75% before the Dataset A recovery, which is almost entirely same-cell.
- both endpoints converged: **26,680** overall, **19,726** same-cell
- F_r-labeled MOVES: **19,726**, of which improving: **6,801** (34.3% of same-cell steps)
- flatness(node_peak)-labeled MOVES: **16,771**, improving: **6,084**
- feasible children (F_r<=1.55, F_q<=2.41, CBC<=1600, |AO|<=0.30, converged): **226** of the same-cell moves, **538** overall (no step has a missing constraint axis)
- elite boards emitted: **444** over **27** cells

## 2. Move-class inference rules

Both boards are decoded with the existing helpers (`lpopt.data.schema.unpack_pattern` -> `lpopt.search.genome.GeneralOrbitGenome.from_pattern`) and diffed at the **orbit-unit** level, which is the level the mutation operators act on. A genome is `(fresh unit -> batch, burned unit -> source unit, centre batch)`. All 29,537 distinct boards in the lineage decoded without a single failure.

| rule (first match wins) | class |
|---|---|
| fresh-unit count +1 / -1 (feed +-4) | `add_fresh_unit` / `remove_fresh_unit` |
| fresh-unit count changed by more | `feed_change_multi` |
| fresh set same size, exactly 1 unit left and 1 joined | `fresh_relocate` |
| fresh set same size, more than 1 swapped in/out | `multi` |
| identical fresh set + wiring, 1 batch label (or centre) changed | `batch_flip` |
| identical fresh set + wiring, 2 labels changed, batch multiset preserved | `batch_swap` |
| identical fresh set + wiring, any other label change | `batch_multi` |
| identical fresh set + labels, exactly 2 burned units exchanged sources | `rewire_swap` |
| identical fresh set + labels, any other wiring change | `rewire_multi` |
| wiring AND labels both changed (fresh set intact) | `multi` |

**Classifier validation (2026-08-15).** Every operator in `lpopt/search/genome.py` was replayed over 300 random genomes each and the classifier recovered the operator name **100% of the time** for all six single moves (`rewire_swap`, `fresh_relocate`, `batch_flip`, `batch_swap`, `add_fresh_unit`, `remove_fresh_unit`).

**But the class is the NET-DIFF class, not the move count — and they come apart.** Campaigns mutate with `n_moves >= 1` composed operators, and a composition can *alias* to a single-move signature: replaying 3-move mutations, 197/400 landed on a net diff indistinguishable from one `fresh_relocate` (relocate A->B then B->C leaves exactly one unit entering and one leaving the fresh set). So `move_class` alone must not be read as "this was one move".

The guard is `n_unit_edits`, and it is sharp because a genuine single move has a bounded edit count:

| move_class | max n_unit_edits for a TRUE single move | corpus median (same-cell) | corpus n |
|---|---|---|---|
| batch_flip | 1 | 1.0 | 1438 |
| batch_swap | 2 | 2.0 | 4 |
| rewire_swap | 2 | 2.0 | 10536 |
| fresh_relocate | 4 | 4.0 | 6021 |

Read that table carefully: `rewire_swap`, `batch_flip` and `batch_swap` sit exactly at their single-move bound in the corpus, so those really are single moves. `fresh_relocate` has a corpus median far above its bound of 4 — **most steps classified `fresh_relocate` are composed moves that merely look like one.** The boolean column `single_move` applies this per-class bound and is the filter to use when training a single-move proposal head; `n_slots_changed` (69-slot Hamming) is carried alongside for the raw board-level distance.

### Where each class lives (same-cell moves vs cross-cell transfers)

| move_class | same_cell | cross_cell |
|---|---|---|
| add_fresh_unit | 0 | 180 |
| batch_flip | 1438 | 0 |
| batch_multi | 6 | 2907 |
| batch_swap | 4 | 0 |
| feed_change_multi | 0 | 3366 |
| fresh_relocate | 6021 | 0 |
| multi | 722 | 0 |
| remove_fresh_unit | 0 | 355 |
| rewire_multi | 170 | 0 |
| rewire_swap | 10536 | 0 |
| sa_unknown | 923 | 830 |

The split is almost perfectly clean and it is a sanity check on the classifier: `feed_change_multi`, `add_fresh_unit`, `remove_fresh_unit` and (cross-pair) `batch_multi` occur **only** on cross-cell edges — a feed morph *must* change the fresh-unit count and a pair transfer *must* relabel batches. Within a cell the campaigns only ever play `fresh_relocate`, `rewire_*`, `batch_*` and composed `multi`, exactly as `construct.py` configures them (`feed_move_prob=0.0`).

Median edit size per class, same-cell moves (unit edits / 69-slot Hamming):

| move_class | n_unit_edits | n_slots_changed |
|---|---|---|
| batch_flip | 1.0 | 1.0 |
| batch_multi | 12.5 | 14.5 |
| batch_swap | 2.0 | 2.5 |
| fresh_relocate | 4.0 | 3.0 |
| multi | 25.0 | 25.0 |
| rewire_multi | 4.0 | 4.0 |
| rewire_swap | 2.0 | 2.0 |
| sa_unknown | 6.0 | 6.0 |

`multi` dominating at ~25 unit edits is the `n_moves_late` composition showing through: those children are not one move away from their parent and are weak supervision for a single-move proposal head.

## 3. Physics annotations — the radial strategy axis

A move-proposal policy has to arbitrate the same trade the engineer argues about out loud: *leakage vs flattening*. Push once/twice-burnt fuel to the periphery (out-in) and you cut neutron leakage — cycle length and CBC go up. Push fresh to the periphery (in-out) and you flatten radial power — F_r and node_peak go down, but you leak. Every step row and every elite row therefore carries the cheap decodable descriptors that make the trade visible.

**Ring definition** — radius terciles of the 69 canonical slots, where `radius = hypot(row, col)` is the vendor `Slot.radius` (0-based quarter-core lattice steps from the centre). Cuts land at r = 5.0000 and r = 7.0711 (= sqrt(50)):

| ring | radius | slots | full_core_assemblies |
|---|---|---|---|
| 0 inner | r < 5.0000 | 22 | 69 |
| 1 middle | 5.0000 <= r < 7.0711 | 23 | 80 |
| 2 peripheral | r >= 7.0711 | 24 | 92 |

All shares are **multiplicity-weighted** (full-core assembly counts, not raw slot counts): a peripheral interior slot is 4 physical assemblies, an axis slot 2, the centre 1.

| descriptor | meaning |
|---|---|
| `fresh_share_inner/middle/periph` | fraction of each ring's assemblies that are FRESH — the in-out vs out-in signature |
| `fresh_r_center` | multiplicity-weighted mean radius of the fresh inventory |
| `fresh_enr_r_center` | the same, additionally weighted by fresh-FA `u_avg_enrichment` — "did this move push REACTIVITY outward or inward?" |
| `once_burnt_periph_share` | fraction of the peripheral ring holding once-burnt FA (chain depth 1) |
| `twice_burnt_periph_share` | fraction of the peripheral ring holding twice-burnt-or-deeper FA (chain depth >= 2) |

Each appears as `parent_*`, `child_*` and `d_*` on every step row, and once per board on every elite row.

Provenance and what is NOT approximated:

- **Residence is exact, not guessed.** The genome's own source-chain resolver (`GeneralOrbitGenome._depths()`) gives every burned orbit unit its chain depth to a fresh root: depth 1 = once-burnt, depth >= 2 = twice-burnt or deeper. Nothing is inferred from burnup.
- **Reactivity weight is `u_avg_enrichment`, NOT `kinf0`.** `data/store/fuel_types.parquet` carries `kinf0`, but it is NaN for 36/194 types including most of the `ga80` library that dominates this lineage — a kinf-weighted descriptor would be silently absent exactly where the corpus is densest. Enrichment is present for every (library_id, type_id) used by any step, so `fresh_enr_r_center` is populated on 100% of steps. If a kinf-weighted variant is wanted later, back-fill `kinf0` for ga80 first.
- **Per-assembly burnup per slot is NOT available** and is not approximated. Residence class (fresh / once / twice+) is the honest resolution the pattern encoding supports; actual slot burnup would need the EDIT5 maps, which exist for only 58% of converged rows.

Direction of the **same-cell moves** on the reactivity-weighted fresh radial centre (`d_fresh_enr_r_center`): outward **3,838**, inward **4,796**, neutral **11,186**. `neutral` is not noise — a `rewire_swap` leaves the fresh set and the batch labels untouched, so its fresh radial centre is *exactly* unchanged and the whole move lives in the burnt inventory. That is why `burnt_periph_dir` (sign of `d_twice_burnt_periph_share`) is carried alongside.

## 4. KEY TABLE — improving fraction by move class and radial direction

This is what the policy has to learn: *which move improves what, where*. `F_r v` / `flat v` / `CBC v` = fraction of labeled steps where the child beat the parent (lower is better); `cyclen ^` = fraction where the child ran longer. Each `n(...)` is that objective's labeled count (both endpoints converged AND both values present), so they differ per column — `flat` and `CBC` are much sparser than `F_r`.

**All tables in this section use the 19,820 SAME-CELL moves only.** Cross-cell transfers are tabulated separately in section 5.

### 4a. By move class

| move_class | n_steps | n(F_r v) | F_r v | n(flat v) | flat v | n(CBC v) | CBC v | n(cyclen ^) | cyclen ^ |
|---|---|---|---|---|---|---|---|---|---|
| rewire_swap | 10536 | 10536 | 0.375 | 9017 | 0.380 | 9067 | 0.440 | 10536 | 0.528 |
| fresh_relocate | 6021 | 5996 | 0.300 | 5116 | 0.355 | 5244 | 0.478 | 5996 | 0.467 |
| batch_flip | 1438 | 1438 | 0.444 | 1263 | 0.406 | 1271 | 0.522 | 1438 | 0.490 |
| sa_unknown | 923 | 923 | 0.298 | 791 | 0.317 | 791 | 0.446 | 923 | 0.515 |
| multi | 722 | 657 | 0.151 | 477 | 0.105 | 657 | 0.437 | 657 | 0.352 |
| rewire_multi | 170 | 166 | 0.253 | 104 | 0.288 | 166 | 0.458 | 166 | 0.512 |
| batch_multi | 6 | 6 | 0.000 | 3 | 0.000 | 6 | 0.333 | 6 | 0.167 |
| batch_swap | 4 | 4 | 0.500 | 0 | n/a | 4 | 0.250 | 4 | 0.250 |

### 4b. By radial direction of the move

`outward` = the move pushed the enrichment-weighted fresh centre toward the periphery (in-out / flattening logic); `inward` = toward the core centre (out-in / low-leakage logic); `neutral` = the move did not touch fresh placement or batch labels at all (pure rewiring of burnt fuel).

| fresh_radial_dir | n_steps | n(F_r v) | F_r v | n(flat v) | flat v | n(CBC v) | CBC v | n(cyclen ^) | cyclen ^ |
|---|---|---|---|---|---|---|---|---|---|
| neutral | 11186 | 11174 | 0.370 | 9485 | 0.377 | 9650 | 0.440 | 11174 | 0.526 |
| inward | 4796 | 4735 | 0.231 | 4068 | 0.262 | 4202 | 0.485 | 4735 | 0.466 |
| outward | 3838 | 3817 | 0.412 | 3218 | 0.448 | 3354 | 0.474 | 3817 | 0.466 |

**The flattening half of the rule of thumb is confirmed.** Moving fresh fuel outward beats moving it inward on both flattening objectives — F_r improves 41.2% of the time going outward vs 23.1% going inward, node_peak 44.8% vs 26.2%.

Correlations over the same-cell, both-converged moves (n=19,726) say the same thing on the continuous descriptor: `d_fresh_share_periph` vs `d_f_r` **-0.131**, vs `d_node_peak` **-0.110** — more peripheral fresh, lower peaking.

**The leakage half does not replicate.** `d_fresh_share_periph` vs `d_cyclen` is **+0.093** and vs `d_cbc_max` **+0.050**: on this corpus, loading fresh outward does NOT cost cycle length. An earlier read of this table on the 1,305-step lpopt-only corpus reported the opposite (outward loading paying a cycle-length bill); that finding **did not survive** the 15x larger corpus and is retracted. The two eras genuinely disagree — see the per-source split in section 4g — and nothing here can say which is right, because neither era sampled direction at fixed move class. This is the single clearest thing the queued 1-move ablation wave has to settle.

The flattening signal, by contrast, holds in both eras and at every sample size tried, so a v1 policy can be trained on it now.

### 4c. By twice-burnt peripheral direction

`outward` = the move pushed twice-burnt FA toward the periphery (low-leakage / out-in logic); `inward` = it pulled them in.

| burnt_periph_dir | n_steps | n(F_r v) | F_r v | n(flat v) | flat v | n(CBC v) | CBC v | n(cyclen ^) | cyclen ^ |
|---|---|---|---|---|---|---|---|---|---|
| neutral | 19746 | 19658 | 0.346 | 16723 | 0.364 | 17138 | 0.458 | 19658 | 0.501 |
| outward | 38 | 34 | 0.088 | 31 | 0.000 | 34 | 0.294 | 34 | 0.441 |
| inward | 36 | 34 | 0.176 | 17 | 0.235 | 34 | 0.647 | 34 | 0.059 |

### 4d. move_class x radial direction — main cell `C1_C2/f121/260624`

The leakage-vs-flattening arbitration the policy has to make, in the densest single cell (groups with >= 5 steps):

| move_class | fresh_radial_dir | n_steps | n(F_r v) | F_r v | n(flat v) | flat v | n(CBC v) | CBC v | n(cyclen ^) | cyclen ^ |
|---|---|---|---|---|---|---|---|---|---|---|
| rewire_swap | neutral | 2021 | 2021 | 0.325 | 2021 | 0.359 | 2021 | 0.429 | 2021 | 0.542 |
| fresh_relocate | inward | 592 | 592 | 0.144 | 592 | 0.260 | 592 | 0.500 | 592 | 0.422 |
| fresh_relocate | outward | 445 | 445 | 0.328 | 445 | 0.438 | 445 | 0.461 | 445 | 0.440 |
| batch_flip | inward | 151 | 151 | 0.450 | 151 | 0.457 | 151 | 0.225 | 151 | 0.775 |
| batch_flip | outward | 147 | 147 | 0.442 | 147 | 0.299 | 147 | 0.864 | 147 | 0.136 |
| sa_unknown | inward | 97 | 97 | 0.175 | 97 | 0.278 | 97 | 0.485 | 97 | 0.515 |
| sa_unknown | outward | 68 | 68 | 0.294 | 68 | 0.265 | 68 | 0.338 | 68 | 0.574 |
| sa_unknown | neutral | 49 | 49 | 0.367 | 49 | 0.367 | 49 | 0.388 | 49 | 0.653 |
| fresh_relocate | neutral | 22 | 22 | 0.364 | 22 | 0.545 | 22 | 0.500 | 22 | 0.318 |

### 4e. Per-cell x move class (cells with >= 100 same-cell moves)

| cell | move_class | n_steps | n_labeled | n_improving | improving_frac |
|---|---|---|---|---|---|
| A01_A02/f121/5.8_5.1 | rewire_swap | 95 | 95 | 47 | 0.495 |
| A01_A02/f121/5.8_5.1 | fresh_relocate | 42 | 42 | 14 | 0.333 |
| A01_A02/f121/5.8_5.1 | sa_unknown | 5 | 5 | 3 | 0.600 |
| A01_A02/f121/5.8_5.1 | batch_flip | 2 | 2 | 1 | 0.500 |
| A01_A04/f121/5.8_5.1 | rewire_swap | 361 | 361 | 105 | 0.291 |
| A01_A04/f121/5.8_5.1 | fresh_relocate | 195 | 195 | 34 | 0.174 |
| A01_A04/f121/5.8_5.1 | sa_unknown | 29 | 29 | 4 | 0.138 |
| A01_A04/f121/5.8_5.1 | batch_flip | 1 | 1 | 1 | 1.000 |
| B1_C2/f121/260624 | rewire_swap | 294 | 294 | 142 | 0.483 |
| B1_C2/f121/260624 | fresh_relocate | 125 | 125 | 54 | 0.432 |
| B1_C2/f121/260624 | batch_flip | 92 | 92 | 48 | 0.522 |
| B1_C2/f121/260624 | sa_unknown | 24 | 24 | 9 | 0.375 |
| B1_C4/f121/260624 | rewire_swap | 65 | 65 | 37 | 0.569 |
| B1_C4/f121/260624 | fresh_relocate | 35 | 35 | 17 | 0.486 |
| B1_C4/f121/260624 | batch_flip | 29 | 29 | 13 | 0.448 |
| B1_C4/f121/260624 | sa_unknown | 14 | 14 | 7 | 0.500 |
| B1_C6/f121/260624 | rewire_swap | 428 | 428 | 188 | 0.439 |
| B1_C6/f121/260624 | fresh_relocate | 218 | 218 | 102 | 0.468 |
| B1_C6/f121/260624 | batch_flip | 110 | 110 | 41 | 0.373 |
| B1_C6/f121/260624 | sa_unknown | 29 | 29 | 14 | 0.483 |
| B3_C6/f121/260624 | rewire_swap | 265 | 265 | 133 | 0.502 |
| B3_C6/f121/260624 | fresh_relocate | 138 | 138 | 56 | 0.406 |
| B3_C6/f121/260624 | batch_flip | 81 | 81 | 29 | 0.358 |
| B3_C6/f121/260624 | sa_unknown | 25 | 25 | 11 | 0.440 |
| B5_C6/f121/260624 | rewire_swap | 88 | 88 | 36 | 0.409 |
| B5_C6/f121/260624 | fresh_relocate | 64 | 64 | 27 | 0.422 |
| B5_C6/f121/260624 | batch_flip | 20 | 20 | 11 | 0.550 |
| B5_C6/f121/260624 | sa_unknown | 12 | 12 | 2 | 0.167 |
| C01_C02/f121/260624 | rewire_swap | 480 | 480 | 211 | 0.440 |
| C01_C02/f121/260624 | fresh_relocate | 252 | 252 | 99 | 0.393 |
| C01_C02/f121/260624 | batch_flip | 97 | 97 | 45 | 0.464 |
| C01_C02/f121/260624 | sa_unknown | 63 | 63 | 18 | 0.286 |
| C01_C04/f121/260624 | rewire_swap | 736 | 736 | 265 | 0.360 |
| C01_C04/f121/260624 | fresh_relocate | 393 | 393 | 121 | 0.308 |
| C01_C04/f121/260624 | sa_unknown | 56 | 56 | 17 | 0.304 |
| C01_C04/f121/260624 | batch_flip | 16 | 16 | 6 | 0.375 |
| C01_C06/f121/260624 | rewire_swap | 87 | 87 | 33 | 0.379 |
| C01_C06/f121/260624 | fresh_relocate | 39 | 39 | 10 | 0.256 |
| C01_C06/f121/260624 | batch_flip | 27 | 27 | 7 | 0.259 |
| C01_C06/f121/260624 | sa_unknown | 5 | 5 | 0 | 0.000 |
| C03_C06/f121/260624 | rewire_swap | 82 | 82 | 39 | 0.476 |
| C03_C06/f121/260624 | fresh_relocate | 31 | 31 | 15 | 0.484 |
| C03_C06/f121/260624 | batch_flip | 15 | 15 | 11 | 0.733 |
| C03_C06/f121/260624 | sa_unknown | 3 | 3 | 1 | 0.333 |
| C1_C2/f121/260624 | rewire_swap | 2021 | 2021 | 657 | 0.325 |
| C1_C2/f121/260624 | fresh_relocate | 1059 | 1059 | 239 | 0.226 |
| C1_C2/f121/260624 | batch_flip | 298 | 298 | 133 | 0.446 |
| C1_C2/f121/260624 | sa_unknown | 214 | 214 | 55 | 0.257 |
| C1_C4/f121/260624 | rewire_swap | 1988 | 1988 | 597 | 0.300 |
| C1_C4/f121/260624 | fresh_relocate | 1133 | 1133 | 245 | 0.216 |
| C1_C4/f121/260624 | batch_flip | 167 | 167 | 63 | 0.377 |
| C1_C4/f121/260624 | sa_unknown | 165 | 165 | 35 | 0.212 |
| C1_C6/f121/260624 | rewire_swap | 1051 | 1051 | 432 | 0.411 |
| C1_C6/f121/260624 | fresh_relocate | 568 | 568 | 207 | 0.364 |
| C1_C6/f121/260624 | batch_flip | 124 | 124 | 52 | 0.419 |
| C1_C6/f121/260624 | sa_unknown | 79 | 79 | 32 | 0.405 |
| C2_C3/f121/260624 | rewire_swap | 208 | 208 | 81 | 0.389 |
| C2_C3/f121/260624 | fresh_relocate | 130 | 130 | 19 | 0.146 |
| C2_C3/f121/260624 | batch_flip | 30 | 30 | 13 | 0.433 |
| C2_C3/f121/260624 | sa_unknown | 19 | 19 | 0 | 0.000 |
| C3_C6/f121/260624 | rewire_swap | 1741 | 1741 | 733 | 0.421 |
| C3_C6/f121/260624 | fresh_relocate | 923 | 923 | 363 | 0.393 |
| C3_C6/f121/260624 | batch_flip | 259 | 259 | 127 | 0.490 |
| C3_C6/f121/260624 | sa_unknown | 139 | 139 | 50 | 0.360 |
| C5_C6/f121/260624 | rewire_swap | 280 | 280 | 107 | 0.382 |
| C5_C6/f121/260624 | fresh_relocate | 166 | 166 | 76 | 0.458 |
| C5_C6/f121/260624 | batch_flip | 35 | 35 | 20 | 0.571 |
| C5_C6/f121/260624 | sa_unknown | 29 | 29 | 13 | 0.448 |
| E1_E2/f117/ga80 | multi | 131 | 113 | 2 | 0.018 |
| E1_E2/f117/ga80 | fresh_relocate | 52 | 49 | 1 | 0.020 |
| E1_E2/f117/ga80 | rewire_multi | 3 | 3 | 0 | 0.000 |
| E1_E2/f117/ga80 | batch_multi | 1 | 1 | 0 | 0.000 |
| E1_E2/f121/ga80 | multi | 63 | 63 | 4 | 0.063 |
| E1_E2/f121/ga80 | fresh_relocate | 61 | 60 | 7 | 0.117 |
| E1_E2/f121/ga80 | rewire_multi | 51 | 51 | 11 | 0.216 |
| E1_E2/f121/ga80 | rewire_swap | 16 | 16 | 4 | 0.250 |
| E1_E2/f121/ga80 | batch_multi | 1 | 1 | 0 | 0.000 |
| E1_E2/f125/ga80 | multi | 114 | 104 | 5 | 0.048 |
| E1_E2/f125/ga80 | fresh_relocate | 59 | 55 | 4 | 0.073 |
| E1_E2/f125/ga80 | rewire_multi | 3 | 3 | 3 | 1.000 |
| E1_E2/f125/ga80 | batch_multi | 1 | 1 | 0 | 0.000 |
| G3_G4/f125/ga80 | multi | 120 | 111 | 12 | 0.108 |
| G3_G4/f125/ga80 | fresh_relocate | 46 | 45 | 3 | 0.067 |
| G3_G4/f125/ga80 | rewire_multi | 1 | 1 | 0 | 0.000 |
| K1_K2/f121/ga80 | rewire_swap | 49 | 49 | 23 | 0.469 |
| K1_K2/f121/ga80 | multi | 27 | 27 | 1 | 0.037 |
| K1_K2/f121/ga80 | fresh_relocate | 20 | 19 | 4 | 0.211 |
| K1_K2/f121/ga80 | rewire_multi | 13 | 13 | 3 | 0.231 |
| K1_K2/f121/ga80 | batch_flip | 8 | 8 | 4 | 0.500 |
| K1_K2/f121/ga80 | batch_swap | 4 | 4 | 2 | 0.500 |
| T6_T4/f121/paramA | multi | 60 | 55 | 5 | 0.091 |
| T6_T4/f121/paramA | fresh_relocate | 52 | 49 | 4 | 0.082 |
| T6_T4/f121/paramA | rewire_multi | 40 | 38 | 9 | 0.237 |
| T6_T4/f121/paramA | rewire_swap | 2 | 2 | 1 | 0.500 |

### 4f. Per-cell x radial direction (cells with >= 100 same-cell moves)

| cell | fresh_radial_dir | n_steps | n(F_r v) | F_r v | n(flat v) | flat v | n(CBC v) | CBC v | n(cyclen ^) | cyclen ^ |
|---|---|---|---|---|---|---|---|---|---|---|
| A01_A02/f121/5.8_5.1 | neutral | 99 | 99 | 0.485 | 99 | 0.414 | 99 | 0.505 | 99 | 0.505 |
| A01_A02/f121/5.8_5.1 | inward | 26 | 26 | 0.269 | 26 | 0.231 | 26 | 0.192 | 26 | 0.615 |
| A01_A02/f121/5.8_5.1 | outward | 19 | 19 | 0.526 | 19 | 0.579 | 19 | 0.632 | 19 | 0.316 |
| A01_A04/f121/5.8_5.1 | neutral | 371 | 371 | 0.286 | 371 | 0.288 | 371 | 0.544 | 371 | 0.431 |
| A01_A04/f121/5.8_5.1 | inward | 134 | 134 | 0.112 | 134 | 0.127 | 134 | 0.201 | 134 | 0.597 |
| A01_A04/f121/5.8_5.1 | outward | 81 | 81 | 0.284 | 81 | 0.321 | 81 | 0.580 | 81 | 0.321 |
| B1_C2/f121/260624 | neutral | 301 | 301 | 0.475 | 301 | 0.478 | 301 | 0.505 | 301 | 0.478 |
| B1_C2/f121/260624 | inward | 127 | 127 | 0.346 | 127 | 0.307 | 127 | 0.504 | 127 | 0.488 |
| B1_C2/f121/260624 | outward | 107 | 107 | 0.617 | 107 | 0.617 | 107 | 0.449 | 107 | 0.589 |
| B1_C4/f121/260624 | neutral | 67 | 67 | 0.567 | 67 | 0.552 | 67 | 0.448 | 67 | 0.478 |
| B1_C4/f121/260624 | inward | 40 | 40 | 0.350 | 40 | 0.325 | 40 | 0.425 | 40 | 0.525 |
| B1_C4/f121/260624 | outward | 36 | 36 | 0.611 | 36 | 0.611 | 36 | 0.528 | 36 | 0.361 |
| B1_C6/f121/260624 | neutral | 437 | 437 | 0.446 | 437 | 0.455 | 437 | 0.481 | 437 | 0.529 |
| B1_C6/f121/260624 | outward | 176 | 176 | 0.534 | 176 | 0.540 | 176 | 0.574 | 176 | 0.511 |
| B1_C6/f121/260624 | inward | 172 | 172 | 0.326 | 172 | 0.291 | 172 | 0.488 | 172 | 0.477 |
| B3_C6/f121/260624 | neutral | 267 | 267 | 0.498 | 267 | 0.498 | 267 | 0.506 | 267 | 0.532 |
| B3_C6/f121/260624 | inward | 133 | 133 | 0.323 | 133 | 0.346 | 133 | 0.564 | 133 | 0.391 |
| B3_C6/f121/260624 | outward | 109 | 109 | 0.486 | 109 | 0.440 | 109 | 0.514 | 109 | 0.523 |
| B5_C6/f121/260624 | neutral | 90 | 90 | 0.411 | 90 | 0.411 | 90 | 0.389 | 90 | 0.567 |
| B5_C6/f121/260624 | inward | 57 | 57 | 0.298 | 57 | 0.298 | 57 | 0.368 | 57 | 0.614 |
| B5_C6/f121/260624 | outward | 37 | 37 | 0.595 | 37 | 0.622 | 37 | 0.297 | 37 | 0.757 |
| C01_C02/f121/260624 | neutral | 503 | 503 | 0.433 | 0 | n/a | 0 | n/a | 503 | 0.529 |
| C01_C02/f121/260624 | inward | 209 | 209 | 0.364 | 0 | n/a | 0 | n/a | 209 | 0.670 |
| C01_C02/f121/260624 | outward | 180 | 180 | 0.439 | 0 | n/a | 0 | n/a | 180 | 0.483 |
| C01_C04/f121/260624 | neutral | 766 | 766 | 0.363 | 0 | n/a | 0 | n/a | 766 | 0.523 |
| C01_C04/f121/260624 | inward | 235 | 235 | 0.226 | 0 | n/a | 0 | n/a | 235 | 0.404 |
| C01_C04/f121/260624 | outward | 200 | 200 | 0.390 | 0 | n/a | 0 | n/a | 200 | 0.460 |
| C01_C06/f121/260624 | neutral | 88 | 88 | 0.375 | 0 | n/a | 0 | n/a | 88 | 0.591 |
| C01_C06/f121/260624 | inward | 41 | 41 | 0.098 | 0 | n/a | 0 | n/a | 41 | 0.390 |
| C01_C06/f121/260624 | outward | 29 | 29 | 0.448 | 0 | n/a | 0 | n/a | 29 | 0.310 |
| C03_C06/f121/260624 | neutral | 83 | 83 | 0.470 | 0 | n/a | 0 | n/a | 83 | 0.434 |
| C03_C06/f121/260624 | outward | 28 | 28 | 0.607 | 0 | n/a | 0 | n/a | 28 | 0.643 |
| C03_C06/f121/260624 | inward | 20 | 20 | 0.500 | 0 | n/a | 0 | n/a | 20 | 0.650 |
| C1_C2/f121/260624 | neutral | 2092 | 2092 | 0.326 | 2092 | 0.361 | 2092 | 0.429 | 2092 | 0.542 |
| C1_C2/f121/260624 | inward | 840 | 840 | 0.202 | 840 | 0.298 | 840 | 0.449 | 840 | 0.496 |
| C1_C2/f121/260624 | outward | 660 | 660 | 0.350 | 660 | 0.389 | 660 | 0.538 | 660 | 0.386 |
| C1_C4/f121/260624 | neutral | 2058 | 2058 | 0.301 | 2058 | 0.347 | 2058 | 0.418 | 2058 | 0.541 |
| C1_C4/f121/260624 | inward | 813 | 813 | 0.149 | 813 | 0.267 | 813 | 0.609 | 813 | 0.363 |
| C1_C4/f121/260624 | outward | 582 | 582 | 0.342 | 582 | 0.450 | 582 | 0.517 | 582 | 0.407 |
| C1_C6/f121/260624 | neutral | 1073 | 1073 | 0.414 | 1073 | 0.392 | 1073 | 0.435 | 1073 | 0.541 |
| C1_C6/f121/260624 | inward | 420 | 420 | 0.283 | 420 | 0.286 | 420 | 0.512 | 420 | 0.471 |
| C1_C6/f121/260624 | outward | 329 | 329 | 0.486 | 329 | 0.486 | 329 | 0.407 | 329 | 0.529 |
| C2_C3/f121/260624 | neutral | 213 | 213 | 0.380 | 213 | 0.366 | 213 | 0.352 | 213 | 0.526 |
| C2_C3/f121/260624 | inward | 97 | 97 | 0.175 | 97 | 0.103 | 97 | 0.567 | 97 | 0.361 |
| C2_C3/f121/260624 | outward | 77 | 77 | 0.195 | 77 | 0.234 | 77 | 0.312 | 77 | 0.558 |
| C3_C6/f121/260624 | neutral | 1798 | 1798 | 0.421 | 1798 | 0.399 | 1798 | 0.436 | 1798 | 0.524 |
| C3_C6/f121/260624 | inward | 659 | 659 | 0.297 | 659 | 0.303 | 659 | 0.481 | 659 | 0.513 |
| C3_C6/f121/260624 | outward | 605 | 605 | 0.529 | 605 | 0.570 | 605 | 0.433 | 605 | 0.577 |
| C5_C6/f121/260624 | neutral | 290 | 290 | 0.386 | 290 | 0.386 | 290 | 0.438 | 290 | 0.538 |
| C5_C6/f121/260624 | inward | 117 | 117 | 0.402 | 117 | 0.359 | 117 | 0.521 | 117 | 0.479 |
| C5_C6/f121/260624 | outward | 103 | 103 | 0.553 | 103 | 0.612 | 103 | 0.282 | 103 | 0.738 |
| E1_E2/f117/ga80 | inward | 101 | 86 | 0.035 | 81 | 0.037 | 86 | 0.349 | 86 | 0.337 |
| E1_E2/f117/ga80 | outward | 61 | 57 | 0.000 | 52 | 0.000 | 57 | 0.404 | 57 | 0.246 |
| E1_E2/f117/ga80 | neutral | 25 | 23 | 0.000 | 21 | 0.000 | 23 | 0.348 | 23 | 0.304 |
| E1_E2/f121/ga80 | neutral | 92 | 92 | 0.185 | 91 | 0.220 | 92 | 0.413 | 92 | 0.500 |
| E1_E2/f121/ga80 | inward | 60 | 59 | 0.051 | 53 | 0.094 | 59 | 0.441 | 59 | 0.593 |
| E1_E2/f121/ga80 | outward | 40 | 40 | 0.150 | 38 | 0.132 | 40 | 0.325 | 40 | 0.325 |
| E1_E2/f125/ga80 | inward | 104 | 93 | 0.032 | 90 | 0.044 | 93 | 0.376 | 93 | 0.355 |
| E1_E2/f125/ga80 | outward | 49 | 47 | 0.064 | 45 | 0.133 | 47 | 0.426 | 47 | 0.426 |
| E1_E2/f125/ga80 | neutral | 24 | 23 | 0.261 | 20 | 0.050 | 23 | 0.348 | 23 | 0.478 |
| G3_G4/f125/ga80 | outward | 75 | 74 | 0.095 | 74 | 0.068 | 74 | 0.392 | 74 | 0.297 |
| G3_G4/f125/ga80 | inward | 71 | 63 | 0.079 | 63 | 0.032 | 63 | 0.270 | 63 | 0.460 |
| G3_G4/f125/ga80 | neutral | 21 | 20 | 0.150 | 19 | 0.105 | 20 | 0.400 | 20 | 0.200 |
| K1_K2/f121/ga80 | neutral | 86 | 86 | 0.384 | 0 | n/a | 86 | 0.500 | 86 | 0.512 |
| K1_K2/f121/ga80 | inward | 21 | 20 | 0.050 | 0 | n/a | 20 | 0.300 | 20 | 0.150 |
| K1_K2/f121/ga80 | outward | 14 | 14 | 0.214 | 0 | n/a | 14 | 0.571 | 14 | 0.143 |
| T6_T4/f121/paramA | inward | 74 | 68 | 0.088 | 68 | 0.088 | 68 | 0.368 | 68 | 0.500 |
| T6_T4/f121/paramA | neutral | 42 | 40 | 0.250 | 40 | 0.250 | 40 | 0.450 | 40 | 0.450 |
| T6_T4/f121/paramA | outward | 38 | 36 | 0.083 | 36 | 0.278 | 36 | 0.389 | 36 | 0.361 |

### 4g. The two eras, side by side

Mean outcome deltas by radial direction, split by lineage source. This is where the disagreement in 4b lives — read the `d_cyclen` column:

| lineage_source | fresh_radial_dir | n | d_f_r | d_node_peak | d_cyclen | d_cbc_max |
|---|---|---|---|---|---|---|
| lpopt_genome | inward | 513 | 0.2137 | 0.2077 | -1.0869 | 13.7162 |
| lpopt_genome | neutral | 383 | 0.0523 | 0.0774 | -0.1564 | 4.8571 |
| lpopt_genome | outward | 409 | 0.0949 | 0.1425 | -2.2099 | 1.0060 |
| sa_mocha | inward | 4222 | 0.0569 | 0.0286 | -0.2455 | -0.9286 |
| sa_mocha | neutral | 10791 | 0.0240 | 0.0169 | 0.0785 | 1.8139 |
| sa_mocha | outward | 3408 | 0.0203 | 0.0095 | 0.1362 | 3.2603 |

`sa_mocha` moves are single MOCHA primitives on 260624/5.8_5.1 boards at feed 121; `lpopt_genome` moves are mostly composed mutations on ga80/paramA boards across many feeds, near the F_r=1.55 boundary. Different libraries, different feeds, different move sizes, different operating points — so the sign flip is not necessarily a contradiction, but it is not resolvable from observational data.

Read 4b/4c/4f as the corpus's own statement of the engineer's rule of thumb, with the caveat that these are **observational** frequencies over whatever the campaigns sampled, not a controlled experiment: the move distribution is not balanced across directions within a class, so a direction effect and a class effect are partly confounded. Section 10 prescribes the ablation wave that would de-confound them.

## 5. Breakdown by campaign and by cell

### 5a. Same-cell moves — top 20 cells

| cell | n_steps | n_labeled | n_improving | improving_frac |
|---|---|---|---|---|
| C1_C2/f121/260624 | 3592 | 3592 | 1084 | 0.302 |
| C1_C4/f121/260624 | 3453 | 3453 | 940 | 0.272 |
| C3_C6/f121/260624 | 3062 | 3062 | 1273 | 0.416 |
| C1_C6/f121/260624 | 1822 | 1822 | 723 | 0.397 |
| C01_C04/f121/260624 | 1201 | 1201 | 409 | 0.341 |
| C01_C02/f121/260624 | 892 | 892 | 373 | 0.418 |
| B1_C6/f121/260624 | 785 | 785 | 345 | 0.439 |
| A01_A04/f121/5.8_5.1 | 586 | 586 | 144 | 0.246 |
| B1_C2/f121/260624 | 535 | 535 | 253 | 0.473 |
| C5_C6/f121/260624 | 510 | 510 | 216 | 0.424 |
| B3_C6/f121/260624 | 509 | 509 | 229 | 0.450 |
| C2_C3/f121/260624 | 387 | 387 | 113 | 0.292 |
| E1_E2/f121/ga80 | 192 | 191 | 26 | 0.136 |
| E1_E2/f117/ga80 | 187 | 166 | 3 | 0.018 |
| B5_C6/f121/260624 | 184 | 184 | 76 | 0.413 |
| E1_E2/f125/ga80 | 177 | 163 | 12 | 0.074 |
| G3_G4/f125/ga80 | 167 | 157 | 15 | 0.096 |
| C01_C06/f121/260624 | 158 | 158 | 50 | 0.316 |
| T6_T4/f121/paramA | 154 | 144 | 19 | 0.132 |
| A01_A02/f121/5.8_5.1 | 144 | 144 | 65 | 0.451 |

### 5b. Same-cell moves — top 20 campaigns

| campaign | n_steps | n_labeled | n_improving | improving_frac |
|---|---|---|---|---|
| 0_Case:sa_2b_cache | 8905 | 8905 | 2793 | 0.314 |
| 0_Case:sa_2b_cache.stale-925b1136 | 6224 | 6224 | 2572 | 0.413 |
| 0_Case:sa_2b_cache.stale-fb857c7a | 2520 | 2520 | 963 | 0.382 |
| 0_Case:sa_2b_cache.stale-b01338df | 760 | 760 | 222 | 0.292 |
| 5-5.25_f117 | 193 | 171 | 12 | 0.070 |
| 5-5.25_f125 | 171 | 157 | 7 | 0.045 |
| 5.5-5.75_f125 | 166 | 156 | 15 | 0.096 |
| 20260713_061541 | 65 | 65 | 32 | 0.492 |
| fpcamp_minfr_T6T4 | 41 | 34 | 12 | 0.353 |
| 5.75-6_f141 | 33 | 27 | 10 | 0.370 |
| 6.25-6.5_f141 | 32 | 31 | 12 | 0.387 |
| 5.5-5.75_f141 | 30 | 26 | 6 | 0.231 |
| fpcamp_minfr_T6T4_r3 | 27 | 26 | 2 | 0.077 |
| fpcamp_minfr_T6T4_r7 | 27 | 27 | 0 | 0.000 |
| fpcamp3_199 | 26 | 25 | 4 | 0.160 |
| 5.25-5.5_f141 | 25 | 19 | 7 | 0.368 |
| fpcamp_minfr2_199 | 25 | 25 | 2 | 0.080 |
| fpcamp_minfr_T6T4_r5 | 25 | 25 | 3 | 0.120 |
| alsearch_G3_G4_f121_frB | 25 | 25 | 3 | 0.120 |
| 6-6.25_f141 | 24 | 23 | 14 | 0.609 |

### 5c. Cross-cell transfers (kept in the parquet, excluded from policy tables)

7,638 edges. What they actually are, by generator:

| generator | n_steps | n_labeled | n_improving | improving_frac |
|---|---|---|---|---|
| elite_perturb | 3501 | 2833 | 1106 | 0.390 |
| n/a | 3345 | 3345 | 937 | 0.280 |
| transfer | 392 | 392 | 224 | 0.571 |
| elite | 278 | 262 | 47 | 0.179 |
| g3_elite_boundary | 122 | 122 | 0 | 0.000 |

`elite_perturb` is `produce.py`'s feed-morph: it takes a store elite, re-seats it onto a different feed stratum (N fresh units changes, so feed changes by 4 per unit) and then mutates. `transfer` carries a good board across fuel pairs, which relabels every fresh batch. Both are genuinely useful physics — they are how the program bootstraps a new cell — but a fixed-cell move policy cannot emit them, and their median edit size (30+ orbit units) confirms they are re-seedings rather than moves. If a *curriculum* / transfer policy is wanted later, this is its training set and it is the larger half of the corpus.

## 6. Lineage coverage — honesty section

### 6a. Dataset A lineage RECOVERED (2026-08-15)

The previous pass reported Dataset A as lineage-free and called it the highest-leverage fix. It has now been done, with zero MASTER cost, by `mine_sa_lineage.py`.

**What the cache encodes, precisely.** The `sa_2b_cache` records themselves carry no lineage — only `key` (the rot61 board) and `rec` (metrics, converged, tag, rule). The lineage lives in the per-run **`sa_log.csv`** that `2_LP/MOCHA/optimizer.py` writes beside the cache, which gives every evaluated candidate its `tag`, its **`move`** (MOCHA's own operator name) and its `accepted` flag.

**It is a PROPOSAL chain, not an accept chain.** The optimizer generates a batch of candidates from one incumbent (`cand, mv = self._move(cur)` for each of `parallel_workers`, all before any acceptance), then applies Metropolis sequentially. So a candidate's parent is the board it was mutated from **whether or not it was accepted** — rejected proposals keep their parent and become negative examples, which is exactly what a move-proposal policy needs. Both readings of "the incumbent" were reconstructed and compared by genome diff size; the batch reading won decisively (median 2.0 unit edits vs 6.0 for the sequential reading), confirming the batch incumbent is the true generative parent. `parent_record_id` uses the batch reading; the sequential one is kept in `sa_lineage.parquet` as `seq_parent_record_id`.

**Tag -> board.** `rec.tag` is per-run and NOT unique across the cache (6,086 tags map to more than one board), so the join goes through each run's own case directories (`runs/<run>/cases/<tag>/cy<NN>/MAS_INP` -> `%LPD_SHF` -> `extract_a.dedup_key_of`), which is the same key `extract_a` used to build the store. The dedup index over all 38,854 Dataset A rows had **0 ambiguous keys**.

**The move classifier was validated against MOCHA's own labels.** This is an independent cross-check the previous pass could not make: `sa_log.csv` says what the move was, and the genome differ says what the boards differ by. They agree exactly where the operator vocabularies overlap (counts from the 4,000-edge `mine_sa_lineage.py --verify` sample):

| MOCHA `move` | lpopt genome class inferred | agreement |
|---|---|---|
| `swap_burned_sources` | `rewire_swap` | 1941/1941 = **100%** |
| `swap_fresh_burned` | `fresh_relocate` | 1026/1026 = **100%** |
| `change_fresh_type` | `batch_flip` (176) / `batch_multi` (527) | see below |
| `compound_shuffle` | spread over 6 classes | marked `sa_unknown` |

`change_fresh_type` splits because MOCHA's operator has **two** strategies (`_mv_change_fresh_type`): repaint one fresh cell (-> `batch_flip`) *or* repaint a whole fuel-type family (-> `batch_multi`, ~30 units at once, and it changes the pair, so those land in the cross-cell bucket). The family repaint is a real MOCHA move with **no counterpart in the lpopt genome vocabulary** — it is described faithfully by its net diff and identified by `source_move`, not forced into a single-operator class.

`compound_shuffle` applies several primitives in one move, so its net diff genuinely is not one operator: those 1,753 steps carry `move_class='sa_unknown'` rather than a fitted label.

**`single_move` is now ground truth for the SA era.** For `lpopt_genome` rows it stays an edit-count inference (a composition can alias to a single-move signature); for `sa_mocha` rows the log names the one operator, so `single_move_evidence` records which basis was used.

**GA lineage was deliberately NOT mined.** `ga_log.csv` exists and carries `operator` + `parents`, but it lists **two** tags even for `clone+mutation`, and `crossover` (2,699 rows) is a genuine two-parent operator that is not a move at all. Rather than guess which listed parent was cloned, its 7,122 rows are excluded and counted here. Resolving them by minimum-diff parent selection is a cheap follow-up worth ~900 extra single-parent steps.

Yield by lineage source (all edges):

| lineage_source | n_steps | n_labeled | n_improving | improving_frac |
|---|---|---|---|---|
| sa_mocha | 21766 | 21766 | 7493 | 0.344 |
| lpopt_genome | 5692 | 4914 | 1622 | 0.330 |

MOCHA's own accept decision vs whether F_r actually improved:

| sa_accepted | n | improving | frac |
|---|---|---|---|
| False | 8717 | 1409 | 0.162 |
| True | 13049 | 6084 | 0.466 |

The two disagree substantially, and they should: MOCHA accepted on a multi-objective Metropolis test over its aggregated `J` (cycle length, boron and peaking together), not on F_r alone. A policy trained to imitate `sa_accepted` would inherit MOCHA's objective; a policy trained on `improved_fr` learns ours. The corpus carries both so the choice stays explicit.

### 6b. What is still missing

- store rows **with** `parent_record_id`: **7,744** / 72,685 (10.7%)
- rows **without** a `parent_record_id` in the store: **64,941**. Dataset A (38,854 rows, 53% of the store) still has the column empty — `lpopt/data/extract_a.py` writes `parent_record_id=None` unconditionally and the store is read-only here — but its lineage is no longer lost: 21,766 of those rows are now reachable through `sa_lineage.parquet` (section 6a). Fixing the column itself belongs in `extract_a.py`, not in this miner.
- children whose parent resolves in the store: **5,692**; **2,052** do not (**1,990** distinct missing parent boards).
- the unresolved ones are not lost data, they are *surrogate-only* boards:
| generator | children | parent_in_store | resolved_frac |
|---|---|---|---|
| elite_perturb | 4274 | 4274 | 1.000 |
| local | 1774 | 21 | 0.012 |
| elite | 1069 | 770 | 0.720 |
| transfer | 392 | 392 | 1.000 |
| g3_elite_boundary | 122 | 122 | 1.000 |
| objective | 60 | 60 | 1.000 |
| proposal | 48 | 48 | 1.000 |
| explore | 5 | 5 | 1.000 |

  `local` (the `_lean_local_search` first-improvement hill-climb in `lpopt/search/campaign.py`) walks the **surrogate** landscape and only its final accepted board is sent to MASTER, so the intermediate `current` boards it names as parents were never evaluated and never entered the store — hence the near-zero resolved fraction above. `elite` children seeded from `prev_top` (previous-wave *predicted* top, `construct.py`) have the same gap.
- audited 2026-08-15 against `data/produce/ledger.jsonl` (58,447 rows): 7,871 of its edges have both endpoints in the store and **0** are new. The store's `parent_record_id` column is therefore complete w.r.t. the produce ledger — there is no extra lineage hiding there.
- audited 2026-08-15 against every `runs/**/labels.jsonl` plus the scratchpad-pulled `t6r*/fpcamp*/minfr*` copies (1,468 records, 1,183 with a parent): **43** records are not in the store and **0** unresolved parent patterns were recoverable from them or from `data/campaigns/**/*.parquet`. The wave label files add nothing the store does not already have.
- both endpoints converged: **26,680** of 5,692 steps (97.2%); the remainder has a non-converged endpoint and therefore an <NA> `improved_fr` label.
- `node_peak` (flatness) is only harvested where an EDIT5 map exists, so **20,579** of 26,680 F_r-labeled steps (77%) also carry a flatness label.
- cyclen band: parsed from a campaign's own deck (`runs/<campaign>/input_deck.inp` or `<campaign>.inp`). Only **15** campaigns ship a deck with `cycle_target_efpd`/`cycle_tolerance_efpd`, covering **83** steps; every other step has `cyclen_band_known=False` and `in_cyclen_band_child=<NA>`. Note the min-F_r campaigns declare their cycle target *report-only* (gates nothing), so the band is informational and is NOT folded into `feasible_child`.
- CBC limit: this corpus uses the program value **1600 ppm** (the `fpcamp_minfr_T6T4` deck header says "CBC gate 1600"), while `lpopt/config.py` still defaults `cbc_limit = 1550.0`. 3,921 converged store rows sit in the 1550-1600 gap and flip feasibility between the two conventions.

## 7. Chain depth — how long are the improvement chains?

Chain length = number of consecutive lineage edges (a depth-1 chain is a single parent->child move). The *improving* chain restricts to edges where the child's F_r beat its parent's. Computed over the same-cell moves, since a cross-cell transfer breaks the action space.

| cell | n_steps | longest_lineage_chain | longest_F_r_improving_chain |
|---|---|---|---|
| C1_C2/f121/260624 | 3592 | 17 | 5 |
| C1_C4/f121/260624 | 3453 | 79 | 5 |
| C3_C6/f121/260624 | 3062 | 25 | 7 |
| C1_C6/f121/260624 | 1822 | 39 | 7 |
| C01_C04/f121/260624 | 1201 | 39 | 5 |
| C01_C02/f121/260624 | 892 | 14 | 4 |
| B1_C6/f121/260624 | 785 | 24 | 7 |
| A01_A04/f121/5.8_5.1 | 586 | 23 | 4 |
| B1_C2/f121/260624 | 535 | 39 | 6 |
| C5_C6/f121/260624 | 510 | 13 | 4 |
| B3_C6/f121/260624 | 509 | 9 | 4 |
| C2_C3/f121/260624 | 387 | 32 | 7 |
| E1_E2/f121/ga80 | 192 | 6 | 1 |
| E1_E2/f117/ga80 | 187 | 2 | 1 |
| B5_C6/f121/260624 | 184 | 7 | 3 |
| E1_E2/f125/ga80 | 177 | 2 | 1 |
| G3_G4/f125/ga80 | 167 | 2 | 1 |
| C01_C06/f121/260624 | 158 | 14 | 3 |
| T6_T4/f121/paramA | 154 | 5 | 2 |
| A01_A02/f121/5.8_5.1 | 144 | 7 | 5 |

Requested focus cells (T6_T4 / E1_E2):

| cell | n_steps | longest_lineage_chain | longest_F_r_improving_chain |
|---|---|---|---|
| E1_E2/f121/ga80 | 192 | 6 | 1 |
| E1_E2/f117/ga80 | 187 | 2 | 1 |
| E1_E2/f125/ga80 | 177 | 2 | 1 |
| T6_T4/f121/paramA | 154 | 5 | 2 |
| E1_E2/f141/ga80 | 3 | 1 | 1 |
| E1_E2/f113/ga80 | 1 | 1 | 0 |

Overall the lineage is **shallow**: the deepest chain anywhere is 79 edges and the deepest strictly-F_r-improving chain is 7. The campaigns re-seed every wave from the store's elite set rather than pushing one board down a long trajectory, so this corpus is a set of **1-step neighbourhoods around good boards**, not a set of long improvement trajectories. A policy trained on it learns *which single move to propose next*; it does not get multi-step credit assignment for free.

## 8. Positional readout — rewire_swap steps

Densest same-cell cell: **C1_C2/f121/260624** (3,592 moves). `rewire_swap` moves there with an F_r label: **2,021**.

Scope: main cell `C1_C2/f121/260624`.

Baseline F_r-improving fraction over this scope: **32.5%**.

By radial SEPARATION of the two rewired orbit units |r1-r2|:

| bin | n | improving_frac |
|---|---|---|
| (-0.001, 1.264] | 678 | 0.353 |
| (1.264, 2.861] | 671 | 0.329 |
| (2.861, 7.602] | 672 | 0.293 |

By mean RADIUS of the two rewired orbit units (r1+r2)/2:

| bin | n | improving_frac |
|---|---|---|
| (1.2060000000000002, 5.05] | 682 | 0.330 |
| (5.05, 6.333] | 667 | 0.286 |
| (6.333, 8.573] | 672 | 0.359 |

Radius is the vendor `ORBIT_UNITS[...].radius` (hypot of the quarter-core row/col of the unit's representative slot), so it is free to compute — no map read, no MASTER call.

## 9. Elite ("good state") set

`data/policy/elites.parquet`: top-20 **feasible** boards per cell, ranked twice — by `f_r` (ascending) and by `node_peak` (ascending) — campaign-blind and wave-blind, exactly the imitation target for a constructor policy. Feasible means converged AND all four program axes known and satisfied; a board missing `cbc_max` is <NA> and is excluded rather than assumed good. Every elite row carries the full physics descriptor set.

The set is small on purpose and the reason is physical: only 1,788 of 64,405 converged boards clear F_r <= 1.55, and 1,272 clear all four axes. F_r is the binding constraint by a wide margin — the program is genuinely running at the licensing boundary, which is exactly why a policy that knows *where to put fuel* is worth more here than a better surrogate.

| cell | f_r | node_peak | total |
|---|---|---|---|
| E1_E2/f121/ga80 | 20 | 20 | 40 |
| J1_J2/f121/ga80 | 20 | 20 | 40 |
| E3_E4/f121/ga80 | 20 | 20 | 40 |
| K3_K4/f121/ga80 | 20 | 20 | 40 |
| T6_T4/f121/paramA | 20 | 20 | 40 |
| J5_J6/f121/ga80 | 20 | 20 | 40 |
| K1_K2/f121/ga80 | 20 | 11 | 31 |
| L1_L2/f121/ga80 | 15 | 5 | 20 |
| N1_N2/f125/ga80 | 12 | 6 | 18 |
| N1_N2/f121/ga80 | 14 | 3 | 17 |
| K1_K2/f125/ga80 | 12 | 5 | 17 |
| J1_J2/f125/ga80 | 13 | 2 | 15 |
| L1_L2/f125/ga80 | 11 | 3 | 14 |
| E1_E2/f125/ga80 | 10 | 4 | 14 |
| G3_G4/f121/ga80 | 9 | 5 | 14 |
| J1_J2/f117/ga80 | 9 | 0 | 9 |
| G3_G4/f125/ga80 | 5 | 4 | 9 |
| E1_E2/f117/ga80 | 4 | 1 | 5 |
| K1_K2/f117/ga80 | 4 | 0 | 4 |
| K5_K6/f121/ga80 | 3 | 0 | 3 |

### 9a. Do our best cores put fresh at the PERIPHERY or INSIDE?

Elite fresh-share-by-ring vs the cell's all-comers average (every converged board in that cell). `d_periph` = elite peripheral fresh share minus all-comers; **positive = the good cores load fresh further OUT (flattening-driven, accepts leakage), negative = further IN (leakage-driven).** Cells with >= 200 all-comers, top 20 by that count.

| cell | n_all | all_inner | all_middle | all_periph | eliteFr_periph | d_periph_eliteFr | eliteFlat_periph | d_periph_eliteFlat |
|---|---|---|---|---|---|---|---|---|
| E1_E2/f121/ga80 | 1097 | 0.383 | 0.448 | 0.638 | 0.707 | 0.068 | 0.739 | 0.101 |
| T6_T4/f121/paramA | 822 | 0.381 | 0.400 | 0.682 | 0.698 | 0.016 | 0.698 | 0.016 |
| E1_E2/f117/ga80 | 501 | 0.407 | 0.479 | 0.550 | 0.576 | 0.027 | 0.609 | 0.059 |
| E1_E2/f125/ga80 | 450 | 0.430 | 0.516 | 0.588 | 0.626 | 0.039 | 0.663 | 0.076 |
| K1_K2/f121/ga80 | 416 | 0.372 | 0.499 | 0.602 | 0.535 | -0.067 | 0.545 | -0.057 |
| G3_G4/f121/ga80 | 406 | 0.444 | 0.521 | 0.530 | 0.556 | 0.026 | 0.583 | 0.053 |
| G3_G4/f125/ga80 | 390 | 0.450 | 0.533 | 0.558 | 0.583 | 0.025 | 0.587 | 0.029 |
| J1_J2/f117/ga80 | 338 | 0.445 | 0.490 | 0.512 | 0.478 | -0.034 | n/a | n/a |
| J5_J6/f121/ga80 | 247 | 0.334 | 0.427 | 0.694 | 0.741 | 0.047 | 0.739 | 0.045 |
| L1_L2/f117/ga80 | 245 | 0.446 | 0.494 | 0.507 | 0.522 | 0.014 | 0.522 | 0.014 |
| N1_N2/f121/ga80 | 230 | 0.449 | 0.522 | 0.525 | 0.537 | 0.013 | 0.594 | 0.069 |
| J1_J2/f125/ga80 | 212 | 0.501 | 0.515 | 0.535 | 0.569 | 0.034 | 0.587 | 0.052 |
| K1_K2/f117/ga80 | 201 | 0.444 | 0.488 | 0.515 | 0.500 | -0.015 | n/a | n/a |

**The answer is OUT, and the flatness-ranked elites are further out still.** Across these cells the F_r-elites sit on average **+0.015** in peripheral fresh share relative to all comers (10/13 cells positive), and the node_peak-elites **+0.042** (10/11 positive). Note also the absolute levels: even the all-comers average loads 0.57 of the peripheral ring fresh against 0.42 of the inner ring, so these cores are already in-out (fresh outside) and the good ones lean further that way. Consistent with section 4b: at F_r 1.55 the binding pressure is radial power flattening, and the corpus pays for it in cycle length.

Two caveats, stated rather than buried. This is an *observational* answer — the elites are whatever the campaigns happened to find, and those campaigns were themselves steered by F_r-minimizing acquisition, so peripheral-fresh boards were preferentially *sampled* as well as preferentially *kept*. And the rule is not universal: `K1_K2/f121/ga80` goes the other way (-0.067). Cells are not interchangeable, which is the whole reason the policy has to be conditioned on the cell rather than taught one global rule.

## 9b. Transpose augmentation — free doubling

`lpopt.data.geometry.transpose` reflects a board across the qi<->qj diagonal. It is an involution, it preserves feed, and it maps every slot to one of **equal radius and equal orbit multiplicity** — so the move class, the unit-edit count, the 69-slot Hamming distance and every ring descriptor are invariant, and the FOMs are the same physical experiment. Only the two pattern strings change.

**Chosen form: an on-the-fly recipe, not materialized rows.** Materializing would write 27,458 duplicate rows across 77 columns to change exactly 2 of them, doubling the parquet for zero new information and creating a second copy to keep in sync. The recipe is three lines at load time:

```python
from lpopt.data.geometry import transpose
from lpopt.data.schema import pack_pattern, unpack_pattern

mirror = steps.copy()
mirror['parent_pattern'] = [pack_pattern(transpose(unpack_pattern(p)))
                            for p in steps['parent_pattern']]
mirror['child_pattern']  = [pack_pattern(transpose(unpack_pattern(p)))
                            for p in steps['child_pattern']]
mirror['augmented'] = True        # every other column is copied verbatim
train = pd.concat([steps.assign(augmented=False), mirror])
```

**Verified, not asserted.** `verify_transpose()` re-derives the class, the edit counts, the Hamming distance and all 7 ring descriptors from the mirrored boards on a random sample of 1,500 steps: **0 invariance violations** (class 0, Hamming 0, physics 0; 0 boards undecodable). Run `python mine_policy_corpus.py --verify-transpose` to reproduce.

Caveat worth stating plainly: this doubles the *training rows*, not the *information*. It teaches the policy the diagonal symmetry it should already respect and it regularizes; it does not add a single new MASTER evaluation. Counting augmented rows toward a data-sufficiency bar would be self-deception, so section 10 counts raw steps and states the augmented figure separately.

## 10. Verdict — is this enough for a supervised move-proposal policy v1?

Bar from the brief: **>= 5,000 labeled steps in the main cell** and **>= 20,000 overall**. Counted over same-cell moves, because a cross-cell transfer is not an action the policy can take.

- main cell `C1_C2/f121/260624`: **3,592** labeled moves — **72%** of the bar.
- overall: **19,726** labeled moves — **99%** of the bar.
- verified SINGLE-move subset (`single_move == True`, i.e. the net diff is one operator AND the edit count proves it): **17,614** labeled moves — **88.1%** of the bar. `rewire_swap` 10536, `fresh_relocate` 5638, `batch_flip` 1438, `batch_swap` 4.
- (counting cross-cell transfers too would give 26,680 labeled edges, 133% of the bar — but they are not moves, so they do not count.)
- with transpose augmentation (section 9b): **39,452** overall / **7,184** main cell. Doubled rows, not doubled information — noted, not banked.

**Verdict: MARGINAL — at the overall bar, short in the main cell.** The Dataset A recovery moved the overall count from 1,305 to 19,726 labeled same-cell moves (15x) and the verified single-move subset from 118 to 17,614 (149x), which is the number that actually matters for a move-proposal head. The remaining gap is concentrated, not diffuse:

1. **No single cell reaches 5,000 labeled moves.** The densest is `C1_C2/f121/260624` at 3,592. The corpus is broad (many cells) rather than deep (one cell), so a per-cell policy is not trainable yet, while a **cell-conditioned** policy over the whole corpus is. That is a modelling choice the data now forces, and it is the right one anyway — section 9a already showed cells are not interchangeable.
2. **The recovered corpus is old-library.** Every `sa_mocha` move is 260624 or 5.8_5.1 at feed 121; the live program runs ga80/paramA across feeds 101-141 near F_r=1.55. The policy will need the cell context as an input and will be extrapolating to the current operating point, so hold out a current-library cell for validation rather than trusting a random split.
3. **The leakage signal is unresolved** (section 4b/4g): the two eras disagree on the sign of the cycle-length response to outward fresh loading. Train the flattening head now; do not train a leakage-arbitration head until the ablation wave lands.
4. **Class imbalance persists at the tails.** The same-cell mix is now `rewire_swap` 10,536, `fresh_relocate` 6,021, `batch_flip` 1,438, `sa_unknown` 923, `multi` 722, `rewire_multi` 170, `batch_multi` 6, `batch_swap` 4. The two structural classes are well covered; `batch_swap` and any within-cell feed move are still effectively absent. A v1 policy can only learn the classes it has seen.
5. **Surrogate-only parents (lpopt era only).** 1,990 distinct parents — mostly the `local` hill-climb, plus `elite`-from-`prev_top` — were never evaluated, so their edges are unusable *as labeled steps*. They remain usable as **unlabeled** move examples for a behaviour-cloning warm start.
6. **The feasible region is tiny.** Only 1,788 of 64,405 converged boards clear F_r <= 1.55 at all; F_r is the binding constraint by a wide margin, which is why the elite set is only 444 rows over 27 cells. The *step* corpus is now healthy; the *good-state* corpus is not, and a constructor policy trained to imitate elites still has very few targets per cell.

Two worries from the previous pass are now **closed**: multi-move contamination (17,614 of 19,726 labeled same-cell moves are verified single moves, most on `sa_log` ground truth rather than an edit-count guess) and flatness sparsity (16,771 same-cell moves now carry a `node_peak` label on both endpoints, up from 870).

**What to do next (cheapest first):**

- **Back-fill `parent_record_id` into the store** from `sa_lineage.parquet` so the lineage is a first-class store column instead of a side file. That is an `extract_a.py` / store-writer change and was deliberately not done here (read-only mandate).
- **Resolve the GA lineage** by minimum-diff parent selection: `ga_log.csv` lists two candidate parents for `clone+mutation`, and the genome differ can say which one the child is one move from. ~900 extra single-parent steps for an hour's work, no MASTER time.
- **Log the local-search chain.** Teach `_lean_local_search` to emit its intermediate boards to the ledger (surrogate-predicted FOMs are fine, flagged as such); every hill-climb step then becomes a step row.
- **Run a dedicated 1-move ablation wave** on the main cell: take the top ~50 elites, apply each move class exhaustively at `n_moves=1`, evaluate. A few thousand MASTER calls buys a *balanced, clean* single-move dataset which is worth more per row than the entire current corpus — and it is the only way to de-confound move class from radial direction, which the observational tables in section 4 cannot do.
- **Stratify that wave by radial direction.** For each elite, propose matched outward / inward / neutral variants of the *same* class so the leakage-vs-flattening trade is measured at fixed move class. That is the experiment that turns the engineer's rule of thumb into a label.
- **Harvest EDIT5 maps on the remaining lineage endpoints.** `node_peak` is the flattening signal and it is still missing on 3,049 of 19,820 same-cell moves (15%) — much better than before, but the gap is what caps the flattening head's training set.
- **Augment by diagonal mirror** at load time (section 9b) — verified label-preserving, free, and it doubles the rows a v1 head sees.
- **Train the flattening head now.** The F_r and node_peak signals are large, consistent across both eras, and backed by 19,726 labeled moves. The leakage-arbitration head waits for the ablation wave.

## 11. Files

| file | rows | what |
|---|---|---|
| `data/policy/steps.parquet` | 27,458 (19,820 same-cell moves + 7,638 cross-cell transfers) | one lineage edge: classified move, physics annotations, outcome labels |
| `data/policy/sa_lineage.parquet` | 21,766 | Dataset A proposal chains recovered from `2_LP/0_Case/runs/*/sa_log.csv` (tag, MOCHA move, accept flag, both parent readings) |
| `mine_sa_lineage.py` | — | the recovery script (read-only over `2_LP`; rerun only if the MOCHA runs change) |
| `data/policy/elites.parquet` | 444 | top-20 feasible boards per cell, by F_r and by node_peak, with radial profiles |
| `data/reports/policy_corpus_20260815.md` | — | this report |

`steps.parquet` columns (77): provenance (`lineage_source`, `source_move`, `sa_accepted`, `single_move_evidence`), context (`campaign`, `dataset`, `generator`, `case_pair`, `feed`, `library_id`, `cell`, `cross_cell`), identity (`parent_record_id`, `child_record_id`, `parent_pattern`, `child_pattern`), diff (`n_slots_changed`, `n_unit_edits`, `move_class`, `single_move`, `swap_span`, `swap_radius`), FOMs (`parent_*`/`child_*`/`d_*` over `f_r`, `cyclen`, `cbc_max`, `f_q`, `ao_abs`, `node_peak`, `map_cov`), physics (`parent_*`/`child_*`/`d_*` over `fresh_share_inner`, `fresh_share_middle`, `fresh_share_periph`, `fresh_r_center`, `fresh_enr_r_center`, `once_burnt_periph_share`, `twice_burnt_periph_share`, plus `fresh_radial_dir`, `burnt_periph_dir`), labels (`improved_fr`, `improved_flat`, `improved_cbc`, `improved_cyclen`, `feasible_parent`, `feasible_child`, `both_converged`, `in_cyclen_band_child`, `cyclen_band_known`).
