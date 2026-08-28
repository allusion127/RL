# Feed-axis production grid — PATHFINDER report

**Date** 2026-08-15 · **Campaign** `feedgrid_pf_20260815` · **Boxes** 198 (probes), 181 (not used — occupied)
**Store at start** `data/store/records.parquet`, 72,685 rows

---

## 0. Headline: the premise was wrong, and that is good news

The task was scoped on the belief that the store is *"~99.7% feed=121"* and that each new
`(pair, feed)` cell would need a seed established from scratch. **It is not.** The store's feed
axis is already broad, and 22 of the 25 target cells already hold converged labels:

```
feed      97   101   105   109   113   117    121   125   129   133   137   141
rows      39  4435   451  5427   600  4706  44417  5312   363  3685   313  2937
```

feed=121 is 61% of rows, not 99.7%. The non-121 mass is dataset **P** — an earlier
`P0_pathfinder` production campaign (`[produce].campaign` in `lpopt.inp` is still set to
`"P0_pathfinder"`) that already walked the feed axis. So this run is not a cold-start
pathfinder; it is a **gap-closing** pathfinder. The budget went to the cells that are
genuinely empty rather than to re-seeding cells that were already seeded.

Three structural findings drove the cell selection:

1. **ga80 tops out at e_core 5.5.** The ga80 fuel table does contain 6.0 anchors
   (`C1..C8`, `data/store/fuel_types.parquet`), but **no ga80 C-pair has ever been run at
   any feed** — zero rows. ga80's realised e_core values are `{5.0 … 5.5}` only. The
   `C1_C2`, `B1_C2`, … pairs that *do* appear in the store are library **`260624`**, a
   different library whose `C` type ids collide by name and sit at e_core ≈ 5.40, not 6.0.
   **The e_core ≈ 6.0 corner of the production grid is reachable only through paramA.**
2. **The paramA feed lattice is step-8.** All ten paramA `P*` pairs share exactly the same
   feed set `{101, 109, 117, 125, 133, 141}` + `121`. **Feeds 113 and 129 are empty for
   every paramA pair** — 10 pairs × 2 feeds = 20 unseeded cells, a whole missing column
   pattern rather than a scattered hole.
3. **The level-1 "promoted" restart cache is dead code.** `CaseAssetResolver.promote()`
   (`lpopt/search/assets.py:867`) has **no callers anywhere in `lpopt/`**. The
   self-improving cache is never populated, `data/produce/promoted/` does not exist, and no
   store row carries a `promoted:` provenance. Every chain re-resolves from level 0/2/3/4
   each time. This campaign measured what that costs (§4): on the one A/B run, populating
   the cache turned 7 chains into 4 for the same 4 labels and rescued 2 patterns that had
   diverged. Flagged as a small durable fix; **deliberately not fixed in this run.**

---

## 1. Seed resolution — every target cell resolves, none needs a bootstrap

`check_single_resolver.py` (the gate that reproduces `produce`'s own single-resolver
selection) on both probe decks:

| deck | library | routing | cells | restart | template | level | provenance |
|---|---|---|---|---|---|---|---|
| `pf815a_ga80.inp` | ga80 | ga80 (paramA routing = False) | 3 | Y | Y | **2** | `pair_feed:*` |
| `pf815b_paramA.inp` | paramA | paramA (paramA routing = True) | 3 | Y | Y | **2** | `pair_feed:*` |

`OVERALL: PASS` — all six cells usable through the single resolver `produce` will build.

**No cy1 bootstrap was required for any cell.** `lpopt/design/bootstrap.py` was not
invoked: level-2 (`pair_feed` — same pair, nearest feed) resolved everywhere, which is the
cheapest fallback that still keeps the pair fixed. Concretely:

- `K1_K2 f129` → `pair_feed:MAS_RST.APRQ_11_0652.86` (the pair's f125 restart)
- `E1_E2 f129` → `pair_feed:MAS_RST.APRQ_11_0635.19`
- `N1_N2 f129` → `pair_feed:MAS_RST.APRQ_11_0677.23`
- `P6257Z1G06N24_P6257Z1G10N12` f113 / f129 / f117 → `pair_feed:MAS_RST.APRQ_11_0740.81`

Two deck-level constraints were re-confirmed and honoured:

- **One resolver per run.** `ProduceDriver._run_library_id` builds exactly one
  `CaseAssetResolver`; a deck naming both libraries routes every paramA stratum through the
  ga80 `FEASIBLE_PACKAGE` and the paramA chains all die. ga80 and paramA therefore ran as
  **two separate decks**, chained in `run_pf815.bat`.
- **Single-cycle discharge.** `allow_single_cycle_discharge` is a `[[produce.strata]]`
  field. It was set explicitly `true` on every f129 stratum. Note it is in fact *auto*-enabled
  at these feeds: `genome.py:753` computes `discharge = allow_single_cycle_discharge or
  n > FRESH_UNIT_COUNT`, and f125→N=31, f129→N=32 both exceed `FRESH_UNIT_COUNT`=30. The
  flag is required at f125/f129 only in the sense that the genome must carry it; the harness
  supplies it itself. It is set explicitly here to document the operating point
  (2·32−60 = 4 unconsumed fresh units per f129 pattern, 0 depth-2 edges).

---

## 2. Per-cell readiness — the 25-cell target grid

`valid ∧ converged` rows already in the store, per cell. `prov` = the fallback level that
seeded it. Cells marked **GAP** had zero rows before this campaign.

| pair | e_core | feed | rows | converged | conv rate | cyclen range (EFPD) | mean cycles | seed provenance | verdict |
|---|---|---|---|---|---|---|---|---|---|
| E1_E2 | 5.00 | 109 | 292 | 177 | 0.606 | 550.5 – 595.2 | 9.16 | pair_feed | READY |
| E1_E2 | 5.00 | 113 | 36 | 36 | 1.000 | 586.8 – 609.1 | 10.31 | pair_feed | READY |
| E1_E2 | 5.00 | 117 | 636 | 501 | 0.788 | 589.0 – 626.2 | 10.85 | **native** | READY |
| E1_E2 | 5.00 | 125 | 592 | 450 | 0.760 | 613.2 – 652.0 | 10.99 | pair_feed | READY |
| E1_E2 | 5.00 | 129 | 16 | 12 | 0.750 | 619.6 – 657.2 | 10.50 | pair_feed | thin → probed |
| K1_K2 | 5.20 | 109 | 34 | 24 | 0.706 | 577.4 – 601.2 | 9.75 | pair_feed | READY (thin) |
| K1_K2 | 5.20 | 113 | 88 | 79 | 0.898 | 588.6 – 633.6 | 9.99 | pair_feed | READY |
| K1_K2 | 5.20 | 117 | 223 | 201 | 0.901 | 607.1 – 648.0 | 9.35 | **native** | READY |
| K1_K2 | 5.20 | 125 | 83 | 67 | 0.807 | 639.9 – 668.4 | 11.18 | pair_feed | READY |
| K1_K2 | 5.20 | 129 | 0 | 0 | — | — | — | **GAP** | probed |
| N1_N2 | 5.40 | 109 | 166 | 122 | 0.735 | 596.9 – 632.3 | 9.43 | pair_feed | READY |
| N1_N2 | 5.40 | 113 | 43 | 32 | 0.744 | 617.5 – 651.0 | 10.28 | pair_feed | READY |
| N1_N2 | 5.40 | 117 | 280 | 217 | 0.775 | 624.0 – 667.8 | 10.67 | pair_feed | READY |
| N1_N2 | 5.40 | 125 | 187 | 150 | 0.802 | 653.5 – 696.9 | 11.11 | pair_feed | READY |
| N1_N2 | 5.40 | 129 | 22 | 14 | 0.636 | 671.6 – 696.4 | 10.71 | pair_feed | thin → probed |
| G3_G4 | 5.50 | 109 | 189 | 128 | 0.677 | 610.5 – 648.0 | 9.38 | pair_feed | READY |
| G3_G4 | 5.50 | 113 | 88 | 67 | 0.761 | 631.7 – 662.6 | 9.55 | pair_feed | READY |
| G3_G4 | 5.50 | 117 | 180 | 144 | 0.800 | 644.4 – 684.6 | 10.39 | pair_feed | READY |
| G3_G4 | 5.50 | 125 | 480 | 390 | 0.812 | 672.1 – 711.7 | 10.76 | pair_feed | READY |
| G3_G4 | 5.50 | 129 | 51 | 38 | 0.745 | 686.9 – 723.7 | 10.45 | pair_feed | READY |
| P6257…G10N12 | 5.94 | 109 | 212 | 156 | 0.736 | 674.9 – 708.9 | 8.60 | pair_feed | READY |
| P6257…G10N12 | 5.94 | 113 | 0 | 0 | — | — | — | **GAP** | probed |
| P6257…G10N12 | 5.94 | 117 | 22 | 18 | 0.818 | 723.3 – 743.9 | 9.33 | **pair_ecore** | thin → probed |
| P6257…G10N12 | 5.94 | 125 | 197 | 164 | 0.832 | 746.1 – 785.4 | 9.73 | pair_feed | READY |
| P6257…G10N12 | 5.94 | 129 | 0 | 0 | — | — | — | **GAP** | probed |

`P6257…G10N12` = `P6257Z1G06N24_P6257Z1G10N12`, the paramA pair closest to e_core 6.0
(5.938). The pairs chosen match the task's suggestion where the library supports it —
E1_E2 (5.0) and K1_K2 (5.2) as suggested; N1_N2 (5.40) is the N-anchor pair; G3_G4 (5.50)
replaces the suggested "K_B cross"; and the ~6.0 slot is a paramA pair, **not** a ga80
C-pair or B_C cross, because ga80 has no realised e_core above 5.5 (§0.1).

### The cyclen/feed trend runs *opposite* to the task's expectation

The brief expected *"lower feed → longer cyclen, e.g. f109 ≈ 700+ EFPD region; f125/129
shorter."* The store says the reverse, monotonically, in every pair:

| pair | f109 | f113 | f117 | f125 | f129 |
|---|---|---|---|---|---|
| E1_E2 | 550–595 | 587–609 | 589–626 | 613–652 | 620–657 |
| N1_N2 | 597–632 | 618–651 | 624–668 | 654–697 | 672–696 |
| G3_G4 | 611–648 | 632–663 | 644–685 | 672–712 | 687–724 |
| P6257…G10N12 | 675–709 | — | 723–744 | 746–785 | — |

**Higher feed → longer cycle**, which is the physically expected direction (more fresh
reactivity loaded per cycle). The ~700 EFPD region is reached at *high* feed and *high*
enrichment, not at f109. Anyone sizing the 625-EFPD target band off the brief's assumption
would have aimed at the wrong corner of the grid.

---

## 3. Probe results — 89 chains, 6 cells, 0 harness errors

Four runs on 198, chained/concurrent, **89 real MASTER chains against a 120 ceiling**:

| run | deck | cells | waves | chains | converged | non-conv | non-finite | harness err |
|---|---|---|---|---|---|---|---|---|
| A | `pf815a_ga80.inp` | 3 ga80 f129 | 9 | 42 | 28 | **0** | 14 | **0** |
| B | `pf815b_paramA.inp` | 3 paramA | 5 | 36 | 32 | 1 | 3 | **0** |
| C1 | `pf815c_fp.inp` | K1_K2 f129 (fallback) | 3 | 7 | 4 | 0 | 3 | **0** |
| C2 | `pf815c2_fp.inp` | K1_K2 f129 (promoted) | 1 | 4 | 4 | 0 | **0** | **0** |
| | | | | **89** | **68** | 1 | 20 | **0** |

Per cell, as merged into the canonical store (`campaign` prefix `feedgrid_pf`):

| pair | e_core | feed | chains | converged | rate | cyclen range (EFPD) | mean cycles | max F_r | s/chain (median) | seed |
|---|---|---|---|---|---|---|---|---|---|---|
| E1_E2 | 5.00 | 129 | 12 | 8 | 0.667 | 626.8 – 659.0 | 10.25 | 2.261 | 299 | `pair_feed` L2 |
| K1_K2 | 5.20 | **129 GAP** | 26 | 18 | 0.692 | 650.6 – 672.3 | 10.78 | 3.135 | 293 | `pair_feed` L2 |
| N1_N2 | 5.40 | 129 | 11 | 8 | 0.727 | 683.6 – 708.2 | 10.38 | 2.504 | 384 | `pair_feed` L2 |
| P6257…G10N12 | 5.93 | **113 GAP** | 14 | 12 | 0.857 | 699.8 – 724.6 | 8.83 | 2.718 | 329 | `pair_feed` L2 |
| P6257…G10N12 | 5.94 | 117 | 8 | 8 | 1.000 | 722.3 – 740.2 | 9.38 | 3.706 | 368 | `pair_feed` L2 |
| P6257…G10N12 | 5.94 | **129 GAP** | 14 | 12 | 0.857 | 754.9 – 794.2 | 9.42 | 3.771 | 409 | `pair_feed` L2 |

**All three true-gap cells converged on the first attempt and hit their full n_target.**
Every measured cyclen fell inside the range predicted from the neighbouring-feed trend in
§2, so the seeds are not merely converging — they are converging to the *right* place.
Zero harness errors across all 89 chains; the single `nonconverged` (paramA f129) and the
20 `non_finite_flux` rows are honest negative labels, not failures of the harness.

---

## 4. Fixed-point identity — **PASS**

The store could not certify this retrospectively: a scan for the same
`(case_pair, feed, pattern)` under two different `restart_provenance` values returns
**0 replicate triples** in all 72,685 rows. So it was measured directly.

Runs C1 and C2 are the same deck, same `random_seed` 8153, same worker count, differing
only in the restart the resolver hands the chain:

- **C1** — `pair_feed:MAS_RST.APRQ_11_0652.86` (level 2, the pair's own **f125** restart:
  a genuine CROSS-FEED fallback).
- **C2** — `promoted:MAS_RST.APRQ_20_0665.22` (level 1, the cell's OWN converged
  equilibrium restart, promoted from a C1 chain via `CaseAssetResolver.promote()`).

All four C2 patterns were drawn identically to C1. Two converged in both runs:

| pattern | cyclen C1 | cyclen C2 | \|Δcyclen\| | \|ΔF_r\| | \|Δcbc\| | cycles C1→C2 |
|---|---|---|---|---|---|---|
| `F:K1:0\|S:1:M:10:2…` | 654.907 | 654.900 | **0.007** | **0.000000** | **0.00** | 10 → 9 |
| `F:K1:0\|S:1:N:14:2…` | 669.854 | 669.854 | **0.000** | **0.000000** | **0.00** | 11 → 11 |

Acceptance bar |Δcyclen| ≤ 2×tol (= 0.8 EFPD) ∧ |ΔF_r| ≤ 2e-3 ∧ |Δcbc| ≤ 2 ppm.
Measured maxima: **0.007 EFPD / 0.000000 / 0.00 ppm** → **PASS**, with two to three orders
of margin. A cross-feed fallback restart lands on the *identical* equilibrium fixed point.

This is corroborated store-wide by the residual of the fixed point itself
(`tolerance_margin`, over all 64,405 converged rows):

| seed level | n | tolerance_margin median | p95 | mean cycles |
|---|---|---|---|---|
| `native` (level 0) | 4,836 | 0.30 | 0.40 | 11.22 |
| `pair_feed` (level 2) | 13,808 | 0.20 | 0.40 | 9.79 |
| `pair_ecore` (level 3) | 6,336 | 0.17 | 0.40 | 9.25 |

The residual distribution is identical across levels, and fallback-seeded chains converge
in *fewer* cycles than native-seeded ones — a nearest-feed restart already sits near the
target equilibrium, whereas a package base restart does not.

### The dead promoted cache is costing labels — flag for a durable fix

The other two C2 patterns are the more consequential result. Both **failed with
`non_finite_flux` from the cross-feed fallback restart and converged cleanly from the
cell-native promoted restart**:

| pattern | C1 (fallback L2) | C2 (promoted L1) |
|---|---|---|
| `F:K1:0\|F:K1:0\|S:1:K:9…` | `non_finite_flux`, no label | **converged, 672.340 EFPD** |
| `F:K1:0\|F:K2:0\|F:K2:0…` | `non_finite_flux`, no label | **converged, 656.986 EFPD** |

Run C1 converged 4 of 7 (0.57); run C2 converged **4 of 4 (1.00)** on the same patterns.
That is consistent with the campaign-wide picture: the f129 column ran 20 `non_finite_flux`
rows out of 89 chains (22%), concentrated in the ga80 f129 cells whose only available seed
is a cross-feed f125 restart four feed-steps away.

The mechanism is clear even though n is small: at the high-feed edge the cross-feed restart
is far enough from the target equilibrium that a fraction of patterns diverge in the early
cycles, and a cell-native restart removes that. **`CaseAssetResolver.promote()`
(`lpopt/search/assets.py:867`) has no callers anywhere in `lpopt/`** — the level-1
self-improving cache the ladder documents is never populated, so every chain pays this
penalty forever.

**Recommended durable fix (small, NOT applied here):** have `ProduceDriver` call
`resolver.promote(case_key, final_restart)` once per cell on the first converged chain.
It is a few lines at the wave-commit site in `lpopt/search/produce.py`, and the resolver
side (atomic write, single-restart normalisation) already exists and is tested by this
run. Note the interaction: `purge_case_dirs` / `purge_intermediate` rmtree the work tree
after **every** wave (`produce.py:1221`), so the promote call must happen *before*
`_maybe_purge()` or there will be no restart left to promote. Left for a separate change
with its own review — deliberately not fixed in this pathfinder run.

---

## 5. Fleet

| box | role | state | used |
|---|---|---|---|
| **198** (`USER@HOST_198`) | probe execution | idle at start — 0 MASTER procs, 32 logical cores | **yes — all probe chains** |
| **181** (`USER@HOST_181`) | second lane | **occupied**: 11 MASTER procs, `mocha_sa_chain_continuation_20260815_0518` started same day | **no** |
| **199** | live r8 campaign | off-limits per brief | no |
| **238** | GPU box | off-limits per brief | no |
| local | 4 MASTER regen procs, 24 cores | light-use only per brief | no MASTER chains run |

Per the budget guard, 181 was left alone because another heavy job owned it; **all probe
chains ran on 198 in a single lane**. Kit reused: `C:\Users\USER\lpopt_work\kit_frontier`
(the existing feed-axis production kit — it already carries `FEASIBLE_PACKAGE`, the paramA
`data/design/package`, and the `venv` with numpy; the system `python` on 198 has no numpy).
Kit code was **not** modified.

Probe artifacts on 198, all new files, no existing file touched:
`pf815a_ga80.inp`, `pf815b_paramA.inp`, `run_pf815.bat`, `data/pf815/` (fresh store dir),
`data/produce/ledger_pf815.jsonl`, `runs_pf815a/`, `runs_pf815b/`.

---

## 6. Cost model for mass production

Per-chain wall time from `data/produce/ledger.jsonl` joined to the store (27,084 chains with
`wall_s`):

| feed | chains measured | median s/chain | mean s/chain |
|---|---|---|---|
| 109 | 5,088 | 303 | 294 |
| 113 | 365 | 214 | 187 |
| 117 | 3,765 | 351 | 332 |
| 121 | 967 | 451 | 424 |
| 125 | 4,956 | 455 | 407 |
| 129 | 363 | 297 | 220 |

paramA is the expensive library (`P6257…G10N12`: 361 s at f109, 487 s at f125, 474 s at
f121) — budget ~450–500 s/chain for paramA cells and ~300–450 s/chain for ga80.

Convergence rate across the grid is **0.61 – 1.00**, clustering at ~0.75–0.80. To land
*N* converged labels, plan **N / 0.75 ≈ 1.33 N** chains.

---

## 7. Store merge — canonical row delta

All probe labels were harvested through the existing produce ledger path and merged with
`python -m lpopt merge-store` (no new label path was invented). Staged as the shape
`_resolve_kit_paths` expects (`<dir>/store/records.parquet` + `<dir>/produce/ledger.jsonl`):

| merge | kit rows | new | upgraded | duplicates | store total | maps | ledger |
|---|---|---|---|---|---|---|---|
| `stage_main` (decks A+B) | 78 | 78 | 0 | 0 | 72,685 → 72,763 | +180 | +156 |
| `stage_c1` (A/B fallback) | 7 | 7 | 0 | 0 | 72,763 → 72,770 | +12 | +14 |
| `stage_c2` (A/B promoted) | 4 | 0 | **2** | **2** | 72,770 → 72,770 | +6 | +2 |
| **total** | **89** | **85** | 2 | 2 | **72,685 → 72,770 (+85)** | **+198** | **+172** |

`stage_c2` merging to 0 new rows is correct and was predicted before running it:
`record_id = sha256(canonical_pattern_string | library_id | case_pair | …)`
(`lpopt/data/schema.py:6`), so run 2's identical patterns carry identical record_ids.
`dedup_upsert` kept the higher-quality row of each pair — which is why the two patterns
that were `non_finite_flux` under the fallback seed now sit in the store as **converged**
labels from the promoted seed. This is also why the A/B had to run into a *separate* store
dir: merging both into one store would have silently collapsed the comparison.

A pre-merge snapshot was taken at `data/store/records.parquet.bak_pre_feedgrid20260815`.

Backfilled campaigns (flagged by merge-store as "not a recognized curriculum cell id" —
expected, this is a pathfinder campaign tag, same convention as the prior `nf_*` cells):
`feedgrid_pf_{E1_E2,K1_K2,N1_N2}_f129`,
`feedgrid_pf_P6257Z1G06N24_P6257Z1G10N12_{f113,f117,f129}`.

---

## 8. READY / NOT-READY verdict — all 25 cells

Every cell in the e_core 5.0–6.0 × feed {109,113,117,125,129} grid now holds converged
labels with a proven level-2 seed. **25 of 25 READY; 0 NOT-READY.**

| pair | e_core | 109 | 113 | 117 | 125 | 129 |
|---|---|---|---|---|---|---|
| E1_E2 | 5.00 | 177 | 36 | 501 | 450 | 20 |
| K1_K2 | 5.20 | 24 | 79 | 201 | 67 | **18 ✱** |
| N1_N2 | 5.40 | 122 | 32 | 217 | 150 | 22 |
| G3_G4 | 5.50 | 128 | 67 | 144 | 390 | 38 |
| P6257…G10N12 | 5.94 | 156 | **12 ✱** | 26 | 164 | **12 ✱** |

✱ = cell created by this campaign (was zero). Converged-label counts.

Readiness rests on three things, all now established for every cell: a seed that resolves
(§1, all level 2, no bootstrap anywhere), chains that converge at a sane rate
(0.61–1.00, campaign probes 0.67–1.00 with zero harness errors), and FOMs that land where
the neighbouring-feed trend predicts (§2, §3).

**Caveat, not a blocker:** the f129 column runs a ~25–33% `non_finite_flux` rate under the
cross-feed fallback seed. It is READY as-is, but §4 shows this is largely an artifact of
the missing level-1 cache. Landing the `promote()` fix before mass production would cut
f129 chain counts materially — on the one A/B measured, 4 chains instead of 7 for the same
4 labels.

---

## 9. Mass-production plan

Convergence rate per cell is measured, so chains = shortfall / rate. Wall-clock uses the
measured per-feed medians of §6 with a ×1.35 paramA premium.

**To 150 labels/cell** — 16 of 25 cells need top-up:

| | |
|---|---|
| chains | **2,007** |
| CPU-hours | **187** |
| wall on **198 alone** @24 lanes | **≈ 7.8 h** |
| wall on **198 @24 + 199 @16** (after r8) | **≈ 4.7 h** |

**To 300 labels/cell** — 22 of 25 cells need top-up:

| | |
|---|---|
| chains | **6,077** |
| CPU-hours | **583** |
| wall on **198 alone** @24 lanes | **≈ 24.3 h** |
| wall on **198 @24 + 199 @16** (after r8) | **≈ 14.6 h** |

Largest single cells to 150 (chains): P6257 f129 161 · P6257 f113 161 · P6257 f117 144 ·
N1_N2 f129 192 · K1_K2 f129 191 · E1_E2 f129 182 · K1_K2 f109 179 · N1_N2 f113 159 ·
G3_G4 f129 151.

Execution notes:

- **Reuse the decks in this run verbatim**; only `n_target` changes. ga80 and paramA must
  stay in separate decks (§1) — the single-resolver rule is not negotiable.
- **199 only after r8 lands.** 181 stays out until its SA campaign finishes.
- 198 ran this campaign at 14 workers on 32 logical cores with room to spare; 24 lanes is
  the realistic sustained figure and is what the estimates above assume. The 199 figure is
  a conservative placeholder (16 lanes) — confirm its core count before committing.
- Land the `promote()` fix first if scheduling allows (§4): it is worth roughly a third of
  the f129 chain budget.

---

## 10. Follow-ups

1. **`resolver.promote()` has no callers** — level-1 cache never populated. Small durable
   fix in `ProduceDriver`, must run before `_maybe_purge()`. Measured benefit: 7 chains → 4
   for the same 4 labels, and 2 diverged patterns rescued. **Not fixed here** (§4).
2. **ga80 e_core > 5.5 is unrealised.** `C1..C8` = 6.0 exist in the fuel table but no ga80
   C-pair has run at any feed. If ga80-native 6.0 is wanted, that is a fresh bootstrap
   (`lpopt/design/bootstrap.py`), not a fallback — no same-pair restart exists to fall back
   to. Today the 6.0 corner is paramA-only.
3. **paramA feed lattice is step-8.** f113/f129 remain empty for the other **9** paramA
   pairs (18 cells). This campaign proved the route works at e_core 5.94; the same level-2
   `pair_feed` seed should open the rest with no bootstrap.
4. **`[produce].campaign` in `lpopt.inp` still reads `"P0_pathfinder"`** — stale, and the
   reason the store's provenance is easy to misread as un-walked. Worth updating.
