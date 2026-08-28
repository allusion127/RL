# batch_swap on the 625 EFPD branch — PRE-REGISTRATION (brief)

**Written 2026-08-15, BEFORE launch.** Third addendum in the series; inherits
cell, descriptors, harness, kit and merge path from
`ablation_wave_prereg_20260815.md` and the deep-sample pattern from
`batchswap_wave_prereg_20260815.md`. Only what is new is stated.

---

## 1. Hypothesis

The first `batch_swap` wave reached **F_r 1.4605**, but on the **~618 EFPD
branch, which is outside the program band**. On the deck's own objective
(`minfr_lambda = 400` EFPD per unit F_r) that board *loses* to the r8 record by
1.68 EFPD-equivalent, and loses to the ga80 incumbent by 14.07. The record is
real and caveated; it is not the number the program can use.

**Registered hypothesis:** the `batch_swap` class transfers to the 625 branch and
pushes the **in-band** frontier below the marks below.

**Stretch target:** an **in-band** board under **1.4636** would be the
unambiguous record — better F_r than the ga80 incumbent *and* inside the band,
so no cyclen caveat attaches.

---

## 2. Marks — and a correction to the brief's mark

The brief names 1.4749 (the r8 campaign record) as the in-band mark. Measured on
the current store, the in-band frontier has **already moved**:

| mark | value | board | cyclen |
|---|---:|---|---:|
| **current in-band best** | **1.4747** | `abd38bc5b212` (my ablation wave) | 625.29 |
| r8 campaign record | 1.4749 | `188c9a338d9f` | 625.46 |
| ga80 incumbent (stretch) | 1.4636 | `deb058c00433` E1_E2 | 633.33 |

So the bar to clear is **1.4747**, not 1.4749 — the ablation wave already took
0.0002 off it. Both are reported; 1.4747 is the one that counts.

**In-band is defined as `cyclen ∈ [620, 645]` EFPD**, per the brief. Registered
caveat, unchanged from the two previous pre-registrations: the deck's own
report-only `cycle_target_efpd = 633.0 ± 5.0` is **not** reconcilable with this
band or with any board in the cell (cell cyclen range 614.66–627.34, so *no* row
would be in-band under the deck). The [620, 645] definition is used here and
gates the readout; the deck discrepancy remains flagged for the campaign owners
and is not resolved by this wave.

Population: 470 feasible boards in the cell, **344 in-band**, 126 out-of-band
(the 1.4605 record among them).

---

## 3. Allocation — 220 chains, at cap, global dedup

Parents are the in-band F_r frontier, deepest first. A `node_peak` parent is
deliberately **excluded**: the registered readout is in-band F_r, and the first
wave showed the record comes from the best parents' neighbourhoods, not from the
weaker parents that have higher improving fractions.

| parent | F_r | cyclen | nbhd | in store | cross-parent dup | take |
|---|---:|---:|---:|---:|---:|---:|
| `abd38bc5b212` | **1.4747** | 625.29 | 224 | 25 | 0 | **70** |
| `188c9a338d9f` | 1.4749 | 625.46 | 224 | 24 | 0 | **70** |
| `1ca37638c03c` | 1.4750 | 623.75 | 224 | 3 | 0 | 40 |
| `a6d11f78a7f6` | 1.4762 | 625.60 | 224 | 5 | **7** | 25 |
| `195699e488e6` | 1.4763 | 623.74 | 224 | 3 | 0 | 15 |

**220 paid, 0 free, all 220 `record_id`s unique.**

**The defect fix from the last wave is applied and already earned its keep.**
Dedup is now GLOBAL — against the store *and* against every earlier parent's
take — rather than per-parent. It caught **7 cross-parent collisions** on
`a6d11f78a7f6`, exactly the failure that wasted 7 chains (3.2% of budget) last
time. Those 7 chains are spent on distinct boards instead.

Within-parent draw: even ranks over `dose`, proportional (unstratified), as
before. Mix: 195 inward / 20 outward / 5 neutral.

**Children are expected to stay in band.** `batch_swap` barely moves cycle
length — measured per-parent median |d_cyclen| in the last wave was 0.02–0.11
EFPD — and every parent here sits at 623.7–625.6, comfortably inside [620, 645].
This is an expectation, not an assumption: in-band status is checked per child
and any that leaves the band is excluded from the §4a readout.

---

## 4. Registered readouts

### 4a. PRIMARY — best **in-band** feasible F_r

Feasible (`mine_policy_corpus.feasibility`) **and** `cyclen ∈ [620, 645]`.
Reported against 1.4747, 1.4749 and 1.4636.

* **in-band best < 1.4636** → the stretch target is met: an unambiguous record,
  better than the incumbent with no cyclen caveat. The strongest available result.
* **in-band best ∈ [1.4636, 1.4747)** → the class transfers to this branch; the
  usable in-band frontier moves; incumbent stands.
* **no in-band improvement on 1.4747** → the class does **not** transfer. The
  618-branch result was branch-specific, and the honest conclusion is that
  `batch_swap`'s value is a property of that branch, not of the operator. Named
  now so it cannot be reframed later.

### 4b. SECONDARY — per-parent improving fraction, deep-n only

Reported **only for parents with n ≥ 40** (`abd38bc5b212` 70, `188c9a338d9f` 70,
`1ca37638c03c` 40). This honours the retraction in
`batchswap_wave_results_20260815.md` §3: n=4 per parent proved to be noise (a
5.8× overestimate on the one parent that could be audited), so small-n parents
are carried in the corpus but **not** quoted as rates. Also reported: full `d_f_r`
distribution per parent, and the inward/outward split at this wave's n.

**Registered expectation:** the first wave measured 0.052 pooled on the 618
branch off parents at 1.4685–1.5117. These parents are tighter (1.4747–1.4763)
and better, so the improving fraction should be **at or below 0.052**. No point
prediction is registered — the last one failed because it transported a rate
across parent sets of different quality, and repeating that error with a
different number would not be a better test.

### 4c. λ-objective comparison vs the r8 record

For the best in-band board: `Δ = −d_F_r × 400 + d_cyclen` EFPD-equivalent against
`188c9a338d9f` (1.4749 @ 625.459). Positive ⇒ the new board wins on the
campaign's own objective. **This is the number that decides whether the result is
usable**, and it is registered as a primary-tier readout precisely because the
last wave's headline F_r record failed it.

### 4d. Corpus

`lineage_source='batchswap_enum_625'` via `ablation_analyze.py corpus`.

---

## 5. Limits

Everything in the two prior pre-registrations still binds. Additionally: one move
class, five parents at 7–31% coverage, best-of-N overstates the class (§4b is the
honest summary), and a null result bounds the class's yield on this branch at
this depth rather than emptying the neighbourhoods. **This still does not reopen
the S1E §8 campaign close-out** — different instrument, as registered twice
before.

---

## 6. Frozen inputs (hashed before launch)

| artefact | sha256 | bytes |
|---|---|---:|
| `batchswap625_wave.py` | `8d981b79523836e7416b80872893def962db7d9012a3a4da7ac62e2b44bf15ba` | 9,545 |
| `data/design/batchswap625_wave_20260815.json` | `0cdca15b6f60c87ed62e3faf05d887bcfbfc67177fc24e3ded0c6264c0e29964` | 475,751 |

Inherited unedited: `ablation_wave.py` (`89d05e83…`), `batchswap_wave.py`
(`56c11457…`). Store at writing: `records.parquet` 73,683 (`abf769f1…`),
`steps.parquet` 27,843 (`8a7e09c8…`). Canonical store read-only until the final
merge. 181 / 198 / 238 untouched. Seed `20260815`; the draw is rank-deterministic.
