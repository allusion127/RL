# A/B round 3 — ADDENDUM: arm R3 (training-data enrichment)

**Written 2026-08-01, BEFORE any R3 interval has been computed.** Addendum to
`data/reports/ab2_preregistration_20260730.md`, inheriting its estimator, cell
unit and non-regression rails unchanged. Only what is stated here is new.
Anything added after the intervals exist is post-hoc and must be labelled.

**Order of work, auditable:** this document was written and saved before
`paired_cell_bootstrap` was called on R3 even once. The only R3 facts known at
the time of writing are the structural comparability checks in §3 — row counts,
`record_id` set/order, column set, split composition, store composition — none
of which involve an R3 prediction value. The verdict lands in
`data/reports/ab2_verdict_R3_20260801.md`.

---

## 1. Why this arm exists

Rounds 1 and 2 killed two of the three mechanisms pre-registration §1 named:

| mechanism (§1) | status |
|---|---|
| loss / label engineering | **falsified** — round 1 (A1, A2, A3), `ab2_verdict_20260731.md` |
| ensemble / variance reduction | **falsified** — E10, `ab2_verdict_E10_20260731.md` |
| **more labels in the deficient cells** | **this arm** |
| verify-many-then-select | untested |

R3 is the leading surviving successor, and the E10 verdict §5 named it as such
before this arm existed. **Admissible.**

R3 = the A0 recipe trained on a **grown store** (69,742 rows vs A0's 67,979)
with a freshly rebuilt S1. `parallel_members=5` is asserted by the coordinator
to be a bit-identity contract, not a recipe change; it is taken as such and is
recorded as an unverified premise in §4.

---

## 2. The two-in-one caveat, stated before any number exists

**R3 changes two things at once and they are inseparable by construction:**

1. new training labels (the +1,763 rows), and
2. a different `data/splits/S1.json`, rebuilt on the grown store.

This is not a design flaw to be apologised for — **data growth IS the arm**, and
a split file that ignored the new rows would not be testing the mechanism. But
it does mean `R3 − A0` is "enrichment + re-split" and cannot be decomposed
further on this evidence. §3 below establishes how much of the re-split half of
that confound actually bites; the answer turns out to be reassuring, but it is
reported as a measurement, not assumed.

### 2.1 The caveat that is NOT reassuring, and that outranks the others

**The enrichment targeted a region that is entirely absent from the decision
surface.** Verified in §3: the frozen 36-cell holdout contains feeds
{101, 109, 117, 125, 133, 141} only and is 100% Dataset P, because
`protect_feed = 121` holds every f121 row in train. The production added its
+782 elite-densification rows and +837 trajectory-coverage rows **in f121
cells**. Therefore:

> **Not one enriched row is in the scoring surface, and no cell of the scoring
> surface was densified.** R3 can only show up on the pre-registered axes through
> *transfer* from f121 into the f101–f141 cells — never directly.

This is registered now, before the result, because it determines how both
outcomes must be read:

* **If R3 improves the target axes:** the effect is transfer, and that is a
  strong result — more labels in one region helping another.
* **If R3 is null on the target axes:** that is **weak evidence about the
  mechanism**, because the mechanism was never given a chance to act locally.
  A null here does **not** falsify "more labels in the deficient cells" the way
  round 1 falsified loss engineering and E10 falsified ensembling. **It falsifies
  only the transfer hypothesis.** Section §6 fixes that reading in advance so it
  cannot be softened or hardened afterwards.

This is a defect in the *experiment's* alignment between mechanism and surface —
the same class of defect round 1 found for A2 (whose provenance offset was inert
on a 100% `master_native` surface). It is the third round in a row where the
pre-registration §5.1 surface has been the limiting factor, and §6 records that.

---

## 3. Comparability — verified BEFORE the rule was fixed

The coordinator asked for this explicitly. All checks are structural.

### 3.1 The row set is identical; the ORDER is not

| check | result |
|---|---|
| `rows_ab2R3.csv` rows | 11,338 (= A0) |
| column set identical to A0 | ✅ |
| `record_id` **set** identical to A0 | ✅ — 0 A0-only, 0 R3-only, 11,338 shared |
| `record_id` **order** identical to A0 | ❌ **differs** |
| duplicate `record_id` in R3 | 0 |

**Growth-invariance of the stable hash held at the row level.** No rows fell
out, so no intersection-pairing and no dropped-row accounting is needed.

**The order difference is handled, not tolerated:** R3 is reindexed onto A0's
`record_id` order before the arena is built, because in this apparatus the row
alignment *is* the pairing (`flat_ab.FlatArena.__post_init__`). A verdict
computed on mis-ordered rows would be silently meaningless, so this is stated
rather than done quietly.

### 3.2 The frozen 36-cell slice did NOT change materially

| property | round 1 / E10 (old S1) | R3 (new S1) | same? |
|---|---|---|:--:|
| cells | 36 | 36 | ✅ |
| cell **names** | (the 36 `e_core_bin×feed` keys) | identical set | ✅ |
| **per-cell row counts** | 90/120/126/119/65/60/… | identical, cell by cell | ✅ **all 36 match exactly** |
| total frozen rows | 3,207 | 3,207 | ✅ |
| frozen rows present in the served CSVs | 3,207 / 3,207 | 3,207 / 3,207 | ✅ |
| feeds on the surface | 101,109,117,125,133,141 | identical | ✅ |
| dataset composition | 100% P | 100% P | ✅ |
| CBC provenance | 100% `master_native` | 100% `master_native` | ✅ |
| CSV `cell` column == S1 cell assignment on frozen rows | ✅ | ✅ | ✅ |

**The frozen slice is unchanged.** The re-split half of the §2 confound is
therefore not biting on the decision surface, and the R3-vs-A0 comparison is as
clean as the round-1 A1/A2/A3-vs-A0 comparisons were.

Residual uncertainty, stated rather than hidden: the *old* `S1.json` has been
overwritten on disk (§3.4), so the old frozen row-ID list cannot be re-read to
prove membership identity directly. What is proved is: identical cell names,
identical per-cell counts, and every new frozen row lies inside the same 11,338
served rows that round 1 verified were all in the old val set. Both val
partitions contain the same 11,338 served rows out of 11,401, so **the two val
sets agree on ≥ 99.45% of their rows and can differ on at most 63 unserved
rows.** A membership swap that preserved all 36 per-cell counts is not
excluded, but it is bounded to nothing that touches the scored data.

### 3.3 Store growth reconciles internally — but not to the itemisation

| quantity | value |
|---|---:|
| store rows, round-1 frozen snapshot | 67,979 |
| store rows now | 69,742 |
| **growth** | **+1,763** |
| train rows, old S1 | 56,578 |
| train rows, new S1 | 58,341 |
| **train growth** | **+1,763** |
| val rows, old S1 / new S1 | 11,401 / 11,401 (unchanged) |

**Every one of the 1,763 new rows went to train; val did not grow by a single
row.** That is exactly what the design wants, and it is internally consistent to
the row.

**Unreconciled:** the coordinator itemises the growth as 782 + 837 + 96 + 113 =
**1,828**, which is **65 more** than the +1,763 the store actually shows. The
likely benign explanation is that 65 of the produced rows were upserts onto
existing `record_id`s rather than new rows, but **this is not verified** and is
recorded as an open discrepancy. It does not affect the verdict — the arm is
whatever was trained — but it means the enrichment count quoted in any
downstream summary should be 1,763, not 1,828.

### 3.4 Snapshot provenance — the frozen inputs are no longer on disk

`data/store/records.parquet` and `data/splits/S1.json` **have been replaced** by
the grown store and the rebuilt split:

| file | pre-registration §2 frozen value | on disk now |
|---|---|---|
| `records.parquet` sha256 | `f01526cc…88b6434` | **replaced** (69,742 rows, 20,577,118 bytes) |
| `records.parquet` rows | 67,979 | 69,742 |
| `S1.json` sha256 | `02d095d8…d2898b99` | **`6ab35f25…ead650640`** |
| `S1.json` train / val | 56,578 / 11,401 | 58,341 / 11,401 |

The round-1 and E10 verdicts recorded those hashes as **verified at the time
they were run**, and they were. They are no longer independently re-runnable
from the repo as it now stands. This is normal for a growing store and is not an
irregularity — but it is recorded here because a reader re-checking
`ab2_verdict_20260731.md` §1 against today's disk will get a mismatch and is
entitled to know why. **A copy of the R3-era store exists at
`store_r3.tgz` (scratchpad); no copy of the round-1-era store or split is known
to exist in this repo.** Preserving the round-1 snapshot would have been the
correct practice and was not done.

---

## 4. Unverified premises

No R3 `meta.json`, `train.log` or model directory exists locally — the only
artifact is `rows_ab2R3.csv`. Assumed on the coordinator's word:

| premise | status |
|---|---|
| R3 = A0 recipe, single change = the grown store + rebuilt S1 | **assumed** |
| `parallel_members=5` is a bit-identity contract, not a recipe change | **assumed** |
| the enrichment composition (+782 elite / +837 traj / +96 panel / +113 campaign) | **assumed, and does not sum to the observed growth** (§3.3) |
| all six per-cell calibrations fitted in one pass (pre-registration §7) | **assumed** |
| R3's five seeds are A0's `20260716..20260720` | **assumed** |

A null result does not depend on these. A favourable result would, and §6 says
so.

---

## 5. The decision rule — fixed before any R3 result exists

### 5.1 Surface, estimator, control — inherited unchanged

Frozen 36-cell holdout, 3,207 rows; unit = cell; control = **A0**, never the
champion; `ab_paired.paired_cell_bootstrap`, `method="bca"`,
`aggregate="median"`, `reps=2000`, `alpha=0.05`, `seed=0`, sign-flipped so
`theta > 0` means "R3 beats A0"; `flat_metrics.cell_mae` at
`min_rows = MIN_CELL_ROWS = 8`; `flat_ab._restrict` to the frozen list.
R3 reindexed onto A0's row order first (§3.1).

### 5.2 Target axes — the deficient-region axes the production aimed at

| axis | metric key | units |
|---|---|---|
| `cbc_max` | `T_cell_mae_cbc_max` | ppm |
| `node_peak` | `T_cell_mae_node_peak` ≡ `M7_cell_mae_node_peak` | — |
| `map_cov` | `M7_cell_mae_map_cov` | — |

`map_cov` cell-MAE has no `T_` key; `M7_cell_mae_map_cov` is the same function
with the same kwargs (as registered for E10 §2.2). As with E10, `node_peak` and
`map_cov` intervals serve both the gain question and the §5.4 rails — one
bootstrap consulted twice, which is how `ab_paired` is designed to be read, not
two tests.

### 5.3 Condition 1 — the bar, and why it is WEAKER than E10's

**R3 clears condition 1 iff BOTH hold:**

1. **at least ONE of the three target axes has `ci_lo > 0`**, and
2. **ZERO of the five variance axes** (`node_peak`, `map_cov`, `f_q`,
   `cbc_max`, `cyclen`) is established worse (`ci_hi < 0`).

E10 had to clear **two** axes; R3 has to clear **one**. That asymmetry is
deliberate and is registered here, with its reason, before the result:

> **E10 doubled serve cost — 5 → 10 forward passes per candidate, campaign wave
> ~20 → ~40 min, permanently. R3's serve cost is UNCHANGED: still 5 members,
> same architecture, same inference path, byte-for-byte the same serve
> contract.** A gain that would not have been worth a 2× standing tax on every
> campaign wave *is* worth having for free. So R3 is allowed the weaker bar, and
> E10 was not, for exactly that reason and no other.

The multiplicity argument that justified E10's two-axis bar still applies (three
axes at α = 0.05 gives ≈ 14% chance of one clearing under a true null), and the
one-axis bar does not answer it. That is an accepted, registered cost of the
weaker bar: **a single-axis R3 pass will be reported as suggestive-but-
multiplicity-exposed, and §5.7's seed check plus the §5.6 secondary observation
are what a reader should weigh alongside it.** Pretending a one-of-three pass is
as strong as a two-of-five pass would be the dishonest move; charging a free arm
E10's price would be the other one.

Clause 2 (zero established-worse) is retained at full strength from E10. More
labels that *degrade* an axis are not "more labels in the deficient cells"
working; they are a distribution shift.

### 5.4 Condition 2 — non-regression rails, inherited verbatim

Every metric in `flat_ab.HARM_MARGINS` must satisfy `harm_upper = −ci_lo < ε`:
`M2_flat_tercile_rho_{node_peak,map_cov}` 0.01; `M3_norm_p_at_8_{node_peak,map_cov}`
0.01; `M5_cell_rho_{f_q,cyclen}` 0.02; `M7_cell_mae_{node_peak,map_cov}` 0.005;
`M7_abs_bias_node_peak` 0.005. `M5_cell_rho_f_r` scored, reported, **excluded
from the verdict** (F_r deferral, `flat_ab.FR_HARM_METRIC`). A straddle is not
equivalence and routes to HOLD.

### 5.5 Verdicts

* **PROMOTE-candidate** — conditions 1 and 2 both hold. Earns a gate attempt
  (pre-registration §5.7); does not promote by itself.
* **HOLD** — condition 1 clause 1 unmet with no axis established worse, and
  condition 2 does not fail.
* **REJECT** — condition 2 fails, **or** any of the five variance axes is
  established worse.
* **ESCALATE** — a target axis returns `insufficient` or `degenerate`.

### 5.6 Registered SECONDARY observation — the targeted region

The production aimed at **E1_E2/f121 and G3_G4/f121**. Per-cell MAE deltas
(`cbc_max`, `node_peak`) on those cells are computed and reported. **This is a
registered secondary observation, not a promotion criterion**, and it cannot
promote R3 unless the §5.3 primary rule passes on its own.

**Its power is pre-emptively disclosed, and it is fatal to the observation as an
inferential test.** Verified in §3 before this rule was written:

| | |
|---|---:|
| served val rows in E1_E2/f121 ∪ G3_G4/f121 | **36** |
| of which `G3_G4` | **0** |
| of which in the frozen 36-cell slice | **0** |
| distinct cell keys they occupy | **1** (`alsearch_E1_E2_f121_minFE`) |

One cell is below `ab_paired.MIN_CELLS = 3`, so a paired cell bootstrap on this
region returns **`method="insufficient"`** by construction — an infinite
interval that fails every gain test and every harm test. **Registered
consequence: the secondary observation will be reported as a row-level
descriptive delta over 36 rows in one cell, carrying NO interval evidence, and
no verdict may cite it.** It is reported because the coordinator asked whether a
large local gain is visible for the frontier campaign, and that question
deserves the number plus an honest statement of what the number cannot support —
not silence, and not a bootstrap dressed up to look like inference.

### 5.7 Power, reproduction, reporting

MDE₈₀ per axis from the same bootstrap (`PairedDiff.mde`, `MDE_POWER = 0.80`);
an effect below its own MDE₈₀ is reported **underpowered**, never null
(pre-registration §5.8). **Any established-worse axis, any harm violation, and
any established gain** is re-run at bootstrap seeds 0–4 and the verdict states
whether it reproduces at every seed — gains included this round, because a
one-axis bar makes a single fragile interval decisive. All intervals, notes,
levels, methods and MDEs go to `data/reports/ab2_verdict_R3_20260801.json`.

### 5.8 Falsification for this addendum

Registered now, and deliberately narrower than rounds 1 and 2:

> **If R3 fails the §5.3 bar, what is falsified is the TRANSFER hypothesis** —
> "labels added in f121 improve the f101–f141 holdout cells" — **not the
> mechanism "more labels in the deficient cells".** The mechanism was never
> exercised on the decision surface (§2.1). The registered consequence of an R3
> null is therefore **not** to abandon label production; it is that the next
> round must **fix the surface/mechanism mismatch** — either hold out a slice of
> the enriched f121 cells so densification can be measured where it happened, or
> move to the last untested §1 successor, verify-many-then-select.

Explicitly forbidden by this addendum, as bigger-versions-of-the-same-knob or as
already-falsified: `--ensemble 20`, larger `--traj-weight`, larger
`--map-peak-topk-weight`, and **another undirected store-growth round scored on
this same f121-free surface** — a second null from the same mismatch would carry
no more information than the first.

---

## 6. Cost

| | A0 / champion | R3 |
|---|---|---|
| members | 5 | **5 (unchanged)** |
| forward passes per candidate | 5 | **5 (unchanged)** |
| campaign inference wave | ~20 min | **~20 min (unchanged)** |
| serve-path code change | — | **none** |
| training cost | ~2.2–2.6 h | ~2.3–2.7 h (one-off, slightly more data) |
| label-production cost | — | 1,763 rows (already spent) |

**R3's serve cost is zero-delta. That is the entire justification for §5.3's
one-axis bar**, and it is why a modest R3 gain would be worth taking where the
same gain from E10 was not.
