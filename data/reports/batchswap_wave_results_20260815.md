# batch_swap deep-sample wave — RESULTS

Pre-registration: `data/reports/batchswap_wave_prereg_20260815.md` (hashed before
launch). One planning defect, one failed registered prediction, and one
unregistered replication are declared in §5–§7.

---

## 1. RESULT

**220 chains run on box 199, 220/220 converged (100%), 0 failures**, 9,196 s
(2.55 h), rc 0. **213 distinct labels** — 7 chains were duplicate boards, a
planning defect of mine (§5).

| | |
|---|---|
| **New cell record** | **F_r 1.4605**, feasible on all four axes — beats the prior record 1.4685 by **−0.0080** and the ga80 incumbent 1.4636 by **−0.0031** |
| Boards beating 1.4685 | **7** |
| Boards beating 1.4636 | **3** (1.4605, 1.4612, 1.4616) |
| **But** | on the campaign's own objective (λ = 400 EFPD per unit F_r) the **r8 board still wins** — §2b. The F_r record is bought with cycle length at a price the deck's own λ rejects. |
| Registered §4b prediction | **FAILED** — measured 0.052, predicted 0.389, naive 0.325. Both outside the CI, and my registered *mechanism* was wrong (§3) |
| Corpus | +213 rows `lineage_source='batchswap_enum'`; +22 r8 back-fill. `steps.parquet` 27,608 → **27,843** |

---

## 2. §4a PRIMARY — best feasible F_r

### 2a. The record

| | |
|---|---|
| record | `4d70ab6f75d4…` **F_r 1.4605** |
| feasibility | F_q 1.8072 · CBC 1302.74 · \|AO\| 0.0239 · converged — clears all four axes |
| cyclen | 618.021 EFPD · node_peak 1.3171 |
| parent | `d84668059508…` (the ablation wave's record, 1.4685) — d_F_r **−0.0080** |

Top of the cell now, all seven from this wave:

| record | F_r | cyclen | CBC | node_peak | parent F_r |
|---|---:|---:|---:|---:|---:|
| `4d70ab6f75d4` | **1.4605** | 618.021 | 1302.74 | 1.3171 | 1.4685 |
| `0a83b6547650` | 1.4612 | 618.793 | 1298.95 | 1.3010 | 1.4877 |
| `456199b370ad` | 1.4616 | 618.094 | 1303.92 | 1.3006 | 1.4685 |
| `eaf914e573b2` | 1.4648 | 618.459 | 1311.89 | 1.2978 | 1.4877 |
| `aa1fe546c549` | 1.4665 | 618.232 | 1289.62 | 1.3076 | 1.4685 |
| `ebc828edb5f9` | 1.4674 | 618.442 | 1312.21 | 1.2987 | 1.4877 |
| `980b4437d7c9` | 1.4683 | 618.742 | 1303.93 | 1.3088 | 1.4685 |

**The whole frontier is now a batch_swap chain off one r4-era elite:**

```
6ccbd209f13b  fpcamp_minfr_T6T4     1.5319  621.53
46e687ed8604  fpcamp_minfr_T6T4_r3  1.5018  621.27
1165441c31ea  fpcamp_minfr_T6T4_r4  1.4877  618.82   <- last campaign-found ancestor
d84668059508  ablation_1move_T6T4   1.4685  618.49   <- batch_swap #1
4d70ab6f75d4  batchswap_enum_T6T4   1.4605  618.02   <- batch_swap #2
```

Two `batch_swap` moves took the cell from **1.4877 → 1.4605 (−0.0272)** — a class
the campaigns played **4 times in 19,820 same-cell corpus moves**. The
under-played-class hypothesis (prereg §1) is confirmed in its **strong form**: one
neglected operator closed a gap four campaign rounds could not.

### 2b. The caveat that must travel with the record

**On the campaign's own objective the new board is not better.** The deck's
`minfr_lambda = 400.0` prices 1.0 unit of F_r at 400 EFPD:

| comparison | d_F_r | worth | d_cyclen | **net** |
|---|---:|---:|---:|---:|
| new `4d70ab6f75d4` vs **r8 record** `188c9a338d9f` (1.4749 @ 625.46) | −0.0144 | +5.76 EFPD-eq | **−7.44** | **−1.68 EFPD-eq → r8 still preferred** |
| new vs **ga80 incumbent** `deb058c00433` (1.4636 @ 633.33) | −0.0031 | +1.24 EFPD-eq | **−15.31** | **−14.07 EFPD-eq → incumbent preferred** |

So: the deck's SECONDARY success criterion is literally `F_r < 1.4636`, and **that
criterion is met**. But the criterion is F_r-only, and the campaign's own
objective function disagrees with it. Reporting "beats the ga80 incumbent"
without this line would be misleading, because the incumbent runs **15.3 EFPD
longer**.

**And the low cyclen is inherited, not caused.** Per-parent median |d_cyclen| for
this wave's moves is 0.02–0.11 EFPD — `batch_swap` barely moves cycle length:

| parent | n | parent cyclen | child cyclen (med) | median d_cyclen |
|---|---:|---:|---:|---:|
| `1165441c31ea` | 70 | 618.815 | 618.796 | −0.020 |
| `d84668059508` | 63 | 618.492 | 618.571 | +0.079 |
| `a4291805f655` | 25 | 618.786 | 618.737 | −0.049 |
| `188c9a338d9f` | 20 | 625.459 | 625.569 | +0.110 |
| `abd38bc5b212` | 25 | 625.285 | 625.387 | +0.102 |
| `c6edd01be332` | 10 | 623.011 | 622.924 | −0.087 |

The cell has **two branches**: a ~618 EFPD branch (where the F_r frontier lives)
and a ~625 EFPD branch (where the campaign optimum lives). This wave went deep on
the 618 branch because that is where the ablation record sat. **The honest
statement is that the F_r frontier and the λ-weighted optimum are on different
branches**, not that one dominates. A λ-weighted push would want this same
enumeration run on the 625 branch — `188c9a338d9f` and `abd38bc5b212` got only 20
and 25 chains here.

---

## 3. §4b SECONDARY — the registered prediction failed, and so did its mechanism

| | value |
|---|---:|
| **measured** (proportional, n=213) | **0.052** [0.022, 0.082] |
| registered prediction (direction-reweighted) | 0.389 — **outside CI** |
| naive balanced pooled (ablation wave) | 0.325 — **outside CI** |

I registered that the ablation wave's balanced design *under*-estimates the true
neighbourhood rate because it over-weights outward, and predicted 0.389 against
0.325. **The measured value is 0.052 — 7.5× below my prediction, and the
direction of my correction was irrelevant.** The registered mechanism is wrong.

**What actually drives it: parent quality, and n=4 being noise.**

Improving fraction tracks how good the parent already is — improving on a better
board is harder:

| parent | parent F_r | n | frac improving | median d_F_r | max d_F_r |
|---|---:|---:|---:|---:|---:|
| `d84668059508` | 1.4685 | 63 | 0.064 | +0.185 | +0.583 |
| `a4291805f655` | 1.4740 | 25 | **0.000** | +0.211 | +0.553 |
| `abd38bc5b212` | 1.4747 | 25 | **0.000** | +0.146 | +0.355 |
| `188c9a338d9f` | 1.4749 | 20 | **0.000** | +0.167 | +0.350 |
| `1165441c31ea` | 1.4877 | 70 | 0.086 | +0.147 | +0.515 |
| `c6edd01be332` | 1.5117 | 10 | 0.100 | +0.121 | +0.320 |

This wave's parents are the frontier (1.4685–1.5117, four of six at ≤ 1.4749);
the ablation wave's `batch_swap` parents were ordinary elites. My prediction
transported a rate across parent sets of very different quality — the error was a
**composition** error, not a direction-weighting error.

**The direct audit on the two shared parents is the sharper result:**

| parent | ablation, balanced n=4 | this wave, proportional | ratio |
|---|---:|---:|---:|
| `1165441c31ea` | **0.500** (2/4) | **0.086** (6/70) | **5.8× overestimate** |
| `188c9a338d9f` | 0.000 (0/4) | 0.000 (0/20) | consistent |

**n=4 per parent is noise.** The ablation wave's per-parent `batch_swap` rates
should not be read as neighbourhood rates at all — a point that applies to every
per-parent cell in ablation results §4, and one the v1.1 fine-tune should not
inherit uncorrected.

**The direction ordering also fails to replicate.** This wave: inward 0.039
(n=178), outward **0.148** (n=27) — outward improves ~3.8× more often. The
ablation wave reported the opposite ordering (inward 0.400 > outward 0.250) on
n=40. At n=213 the ordering reverses, and it now agrees with the corpus's
flattening rule. **The ablation wave's `batch_swap` improving-fraction split was
under-powered and should be retracted**; its *paired* leakage statistics, which
are a different estimand, are unaffected and replicate (§4).

---

## 4. Unregistered replication — the leakage sign holds at 5× the sample

Declared as unregistered: this wave happens to be 213 `batch_swap` moves, so the
ablation wave's secondary leakage instrument (n=40) can be re-run.

| statistic | ablation (n=40) | this wave (n=213) |
|---|---:|---:|
| mean(out−in) `d_cyclen` | −0.1128 [−0.1995, −0.0262] | **−0.1893 [−0.3029, −0.0756]** |
| mean(out−in) `d_f_r` | −0.1711 | **−0.1442 [−0.1888, −0.1013]** |
| dose-response slope | −9.41 [−15.53, −4.46] | **−21.19 [−25.88, −18.83]** |

**The sign replicates on every statistic, and the CIs exclude zero.** Outward
still costs cycle length and still flattens. **The slope magnitude does not
replicate** — the two CIs do not overlap (−9.4 vs −21.2). So the ablation wave's
headline direction verdict is robust, but its *quantitative* dose coefficient is
parent-set-dependent and should be quoted as a sign with an order of magnitude,
not as a calibrated EFPD-per-unit constant. The sign test here is only 0/4
(the proportional draw left few parents with both directions well-sampled); the
bootstrap CI carries the result.

---

## 5. Defect — 7 duplicate chains (3.2% of budget)

The plan deduplicated candidates **within** each parent but not **across**
parents. `d84668059508` *is* `1165441c31ea` with units 48↔51 exchanged, so a
`batch_swap` touching 48 or 51 on one lands on a board also reachable from the
other — e.g. `d84…` `bs:36<->48` and `1165…` `bs:36<->51` are the same board.
Seven such collisions were each evaluated twice: **220 chains, 213 distinct
labels, ~35 min of MASTER wasted.**

The store merge handled it correctly (213 new / 0 duplicate, 7 recognised as
already present within the kit). The fix for any future wave is a global
`record_id` dedup across the whole candidate pool, not per parent.

**Silver lining — a free determinism check.** All 7 collided pairs were run as
independent chains and returned **bit-identical F_r** (1.4612, 1.4838, 1.6427,
1.6999, 1.6955, 1.4897, 1.4648 — 7/7 exact). MASTER is deterministic in this
cell at this restart, confirmed on 7 independent pairs rather than assumed.

---

## 6. Corpus rows added

| | before | after |
|---|---:|---:|
| `data/policy/steps.parquet` | 27,608 | **27,843** |
| ↳ `batchswap_enum` | — | **213** |
| ↳ r8 back-fill (`lpopt_genome`) | — | **22** |
| `data/store/records.parquet` | 73,470 | **73,683** (+213) |
| `data/store/maps.npz` | — | +213 |

All 213 are `single_move=True`, `move_class='batch_swap'` (178 inward / 27
outward / 8 neutral). The **r8 gap flagged in ablation results §6 is closed** —
22 edges appended with their correct native `lineage_source='lpopt_genome'`
(they are ordinary campaign edges, not interventional), so the T6_T4 corpus
series is now complete: base, r3–r8, frtransfer, ablation, batchswap.

Appender: `ablation_analyze.py corpus --campaign … --lineage …`, parameterised
rather than duplicated; it calls `mine_policy_corpus.build_steps` itself.
`ablation_wave.py` remains unedited (hash `89d05e83…` still valid);
`batchswap_wave.py` rebinds only its two provenance constants.

The 198 feed-grid tranche merged by the coordinator mid-wave (store 73,020 →
73,470) touched **none** of this cell — verified before analysis: 1,220 rows /
425 feasible / best 1.4685 unchanged, so every registered comparison mark stands.

---

## 7. Limits

Prereg §5 still binds. Additionally:

* **Best-of-N overstates the class.** 213 draws produced 7 boards under 1.4685;
  the *median* `batch_swap` degrades F_r by ~+0.15. §3 is the honest summary of
  the class, not §2.
* **Two branches, one explored.** 158 of 213 chains went to ~618 EFPD parents.
  Nothing here says what `batch_swap` does on the 625 EFPD branch at depth.
* **Coverage is 4–31% per parent**; a zero improving fraction on three parents
  bounds their yield at this depth, it does not empty their neighbourhoods.
* **This still does not reopen the S1E §8 close-out** — different instrument, as
  registered. The close-out concerned guided-search campaign rounds; nothing here
  argues for restarting them.

---

## 8. Artefacts

| path | what |
|---|---|
| `data/reports/batchswap_wave_prereg_20260815.md` | pre-registration |
| `data/reports/batchswap_wave_results_20260815.md` | this report |
| `data/reports/batchswap_wave_tables.txt` | raw readout |
| `batchswap_wave.py` | plan/run/kit — sha256 `56c11457…` |
| `batchswap_analyze.py` | §4a/§4b readouts |
| `data/design/batchswap_wave_20260815.json` | plan, sha256 `bb579a6a…` |
| `runs/batchswap_enum_T6T4/ablation_results.jsonl` | 220 raw outcomes, sha256 `4c29301d…` |
| `runs/batchswap_enum_T6T4/kitdata/` | merged kit (213 rows, 213 maps) |

Post-merge: `records.parquet` `abf769f1…`, `steps.parquet` `8a7e09c8…`.
Backups: `steps.parquet.bak_pre_batchswap_enum_T6T4`,
`steps.parquet.bak_pre_fpcamp_minfr_T6T4_r8`.

199 is free (`master=0`, rc 0). 181 / 198 / 238 untouched.

---

## 9. What this changes

1. **A `batch_swap` pass belongs in the campaign operator mix.** Two moves of a
   4-in-19,820 class beat four campaign rounds on F_r. The cheapest fix is
   raising `batch_prob` or adding an explicit swap sweep at wave end.
2. **Run the same sweep on the 625 EFPD branch** before treating 1.4605 as the
   cell's answer — the λ-weighted optimum lives there and got 45 chains.
3. **Reweight or discard the ablation wave's per-parent `batch_swap` rates**
   (§3): n=4 is noise, and its direction ordering for this class is retracted.
4. **Quote the leakage dose coefficient as a sign, not a constant** (§4).
5. **Global record_id dedup in the next plan** (§5).
