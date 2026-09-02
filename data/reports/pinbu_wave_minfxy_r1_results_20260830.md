# Phase-2 pin-burnup + F_xy determinism wave — RESULTS (min_fxy T6_T4/f121 r1, 2026-08-30)

**Pre-registration** `data/reports/pinbu_wave_minfxy_r1_prereg_20260830.md` (binding, incl. its three STAMP appendices)
**Manifest** `data/reports/pinbu_wave_minfxy_r1_manifest.json` · **Run dir** `runs/pinbu_wave_minfxy_r1`
**Deck** `pinbu_wave_minfxy_r1_199.inp` · **Box** HOST_199 · **Harness** `pinbu_wave.py run` / `patch`
**Offline `F_xy` scan** `data/reports/fxy_backfill_199_pinbu_wave_minfxy_r1_20260830.csv` (25 rows, 25 sane, `cycle_evidence = final`)
**Scored against the prereg as written.** No deck, harness or analysis code was changed for this readout.

---

## 0. Marks

| mark | registered prediction | measured | verdict |
|---|---|---|---|
| **M1** pin on `minfxy_r1_top20` | **20/20 ≤ 80 AND 20/20 ≤ 78** | **20/20 ≤ 80, 20/20 ≤ 78**; span 62.451–63.760 | **PASS** |
| **M2** \|ΔF_xy\| on the 20 replays | **≤ 0.002, max over 20** | **0.000000 on 20/20** (exact) | **PASS** |
| **M3** chain integrity | 25/25 converged, 25/25 `provenance_ok` | **25/25 converged, 25/25 `provenance_ok`**, 25/25 `determinism_ok` | **PASS** |
| **M4** `F_xy` of the `F_r`-record family | **all five > 1.5322**; `4d70ab6f` ≈ 1.59 | **all five > 1.5322** (1.5402–1.5684); `4d70ab6f` = **1.5402** | **PASS** (level), **point estimate missed** |
| M5 feed-121 pin-head bias | replicate `+3.230`, CI [+2.39, +4.07] | **+3.420**, MAE 3.420, sd 0.485, n=25, CI **[+3.23, +3.61]** — **overlaps** | reported — replicated |
| M6 scalar determinism | ≤ precedent (0 / 1e-4 / 0.02 / 0.006 / 1e-4 / 0.003) | **0.000000 on all six axes** | reported — better than precedent |
| M7 `F_xy`/`F_r` ratio below `F_r` 1.4769 | group-A stored span 1.0158–1.0697, median 1.0640 | group B **1.0546–1.0634**, median 1.0588 — inside the span, all ≤ the median | reported — `S1` wall still not scorable |
| M8 `f_xy` head level skill | bias −0.0075, MAE 0.0238 vs `G2′ MAE < 0.0767` | confirmed **exactly** (measured = stored on 20/20); clears `G2′` on this cell | reported |

Wave wall **1,262 s (0.35 h)** against the ~0.35 h expectation and the **1.0 h registered cap** — **inside budget**, and the linear scaling from the precedent was exact. Per-chain wall median 328 s, max 596 s.

**Headline.** The r1 optimum `bf3a70b2` measures **63.760 GWd/tU** pin — 16.2 GWd/tU under the LEU+ limit — and is now **`is_deliverable = True` on all six axes**. All 20 top-20 cores are deliverable-grade, and so are all 5 `F_r`-era cores. **The store went from 0 to 25 joint-clean + deliverable rows**; this wave created the entire deliverable population.

**Counter-headline (M4, decision-grade).** The `F_r` record core `4d70ab6f` measures `F_xy` **1.5402** — it does not beat 1.5322, so M4 is not falsified, but it sits **only +0.0080 above** the r1 optimum and **below** the 1.5491 incumbent r1 reported beating. The cell's true pre-r1 best `F_xy` was 1.5402, not 1.5491, so **r1's real gain is 0.0080 (−0.52%), not the headline 0.0169 (−1.09%)**. See §5.

---

## 1. Run integrity (M3) — **PASS**

| chains | converged | determinism ok | provenance ok | pin present | **usable** |
|---:|---:|---:|---:|---:|---:|
| 25/25 | 25 | 25 | **25** | 25 | **25** |

The registered 25/25 provenance expectation holds. As prereg §7 argued, this is **structural, not lucky**: all 25 targets were `native:MAS_RST.APRQ_10_0615.11`, resolved at plan time at `fallback_level = 0`, on a restart that never resolves out of `data\produce\promoted`. The precedent's 8/40 `pair_*` drift is **absent**, not mitigated. The promoted-cache cell-count gate (`$wantPromCells = 8`, stamped) held.

Determinism control (measured − stored, all 25 chains):

| axis | n | max abs delta | tolerance | precedent |
|---|---:|---:|---:|---:|
| `f_r` | 25 | **0.000000** | 0.002 | 0.000000 |
| `cyclen` | 25 | **0.000000** | 0.5 | 0.006000 |
| `cbc_max` | 25 | **0.000000** | 2.0 | 0.020000 |
| `f_q` | 25 | **0.000000** | — | 0.000100 |
| `ao_abs` | 25 | **0.000000** | — | 0.000100 |
| `max_assembly_burnup` | 25 | **0.000000** | — | 0.003000 |

**M6 verdict: bit-exact on all six axes, 25/25** — strictly better than the precedent, which was exact only on `f_r`. On one restart, one library and one cell, MASTER's equilibrium is reproducible to the printed precision.

---

## 2. M1 — pin axial peak on the r1 `F_xy` frontier → **PASS**

**20 of 20 measure ≤ 80.0 GWd/tU (100%)** and **20 of 20 ≤ 78.0 (100%)**, both exactly as registered. Measured span **62.451 – 63.760**, median 63.228 — a 16.2 GWd/tU margin at the worst core.

> **The number the wave was run for: rank 1 `bf3a70b2` measures `max_pin_burnup` = 63.760 GWd/tU.**
> Its `unknown_axes = ("max_pin_burnup",)` is closed. See §6.

| rank | record | stored `F_xy` | `F_r` | pred pin | **meas pin** | pred−meas | ≤80 | ≤78 |
|---:|---|---:|---:|---:|---:|---:|:-:|:-:|
| 1 | `bf3a70b20e50` | **1.5322** | 1.4857 | 66.59 | **63.760** | +2.83 | ✔ | ✔ |
| 2 | `d06a56d6b928` | 1.5337 | 1.5098 | 67.77 | **63.183** | +4.59 | ✔ | ✔ |
| 3 | `8b5221442ce8` | 1.5490 | 1.4912 | 66.50 | **63.133** | +3.37 | ✔ | ✔ |
| 4 | `e1495cbe169e` | 1.5577 | 1.5076 | 66.58 | **63.386** | +3.19 | ✔ | ✔ |
| 5 | `2613a46a257a` | 1.5631 | 1.4966 | 66.68 | **63.615** | +3.06 | ✔ | ✔ |
| 6 | `cbed1e74dee6` | 1.5689 | 1.5210 | 66.49 | **63.465** | +3.02 | ✔ | ✔ |
| 7 | `89a971f00758` | 1.5731 | 1.5238 | 66.63 | **62.914** | +3.71 | ✔ | ✔ |
| 8 | `70dbb5fea765` | 1.5777 | 1.4769 | 66.65 | **63.278** | +3.37 | ✔ | ✔ |
| 9 | `3a76842799f0` | 1.5783 | 1.4867 | 66.51 | **62.736** | +3.78 | ✔ | ✔ |
| 10 | `50f593b80bdf` | 1.5783 | 1.4780 | 66.52 | **63.421** | +3.10 | ✔ | ✔ |
| 11 | `3615e5f5bcd6` | 1.5817 | 1.4791 | 66.52 | **63.312** | +3.21 | ✔ | ✔ |
| 12 | `016c7fd4e827` | 1.5819 | 1.4788 | 66.53 | **62.612** | +3.92 | ✔ | ✔ |
| 13 | `0dd07dd42c1d` | 1.5832 | 1.4826 | 66.73 | **63.273** | +3.46 | ✔ | ✔ |
| 14 | `98acafa3e63d` | 1.5839 | 1.4830 | 66.68 | **63.276** | +3.40 | ✔ | ✔ |
| 15 | `a4123b988e5a` | 1.5842 | 1.4826 | 66.82 | **62.603** | +4.22 | ✔ | ✔ |
| 16 | `094c7629068e` | 1.5871 | 1.4911 | 66.49 | **62.558** | +3.94 | ✔ | ✔ |
| 17 | `b456608c8ef2` | 1.5876 | 1.4927 | 66.41 | **63.060** | +3.35 | ✔ | ✔ |
| 18 | `de57276dc848` | 1.5883 | 1.4879 | 66.67 | **63.503** | +3.17 | ✔ | ✔ |
| 19 | `e591bddf9139` | 1.5883 | 1.4919 | 66.51 | **62.771** | +3.74 | ✔ | ✔ |
| 20 | `e88e7be53e5c` | 1.5890 | 1.5101 | 66.73 | **62.451** | +4.28 | ✔ | ✔ |

The registered basis held on every clause: predicted 66.41–67.77, measured 62.45–63.76, **no negative residual anywhere (0/25)**, and the worst residual (+4.59) is well inside the "even −5.0 would still clear" argument. The prereg's expectation of "measured values near 63–65" landed on the low edge of its own band.

### 2.1 The precedent's feed split is confirmed, and this is the clean feed

`pinbu_wave_fxyera_r1` §2.1 found feed 121 cleared the limit 10/13 with a **+4.78** head bias, and feed 109 was 0/7. This wave puts **25 feed-121 chains** in the same place and gets **25/25 ≤ 78** with bias +3.42. The two waves agree: **at feed 121 pin burnup is an advisory column, not a live constraint**, and the `minfxy_pin_bu_limit = 78` acquisition gate costs this cell nothing.

Across the 25 measured cores, pin correlates with **assembly burnup** (`r = +0.752`, pin/max-assy ratio a tight 1.1412–1.1549) and **negatively** with `F_xy` (`r = −0.606`) and `F_r` (`r = −0.375`). The precedent's finding that low `F_xy` neither buys nor costs pin margin holds; here, if anything, the flatter cores run *hotter* on pin — a within-cell burnup effect (flatter power ⇒ more uniform depletion ⇒ slightly higher assembly burnup), not an `F_xy` mechanism.

---

## 3. M2 — `F_xy` determinism → **PASS (exact)**

All 20 `minfxy_r1_top20` chains reproduce their stored `f_r` / `cyclen` / `cbc_max` inside tolerance (in fact to 0.000000) and keep their `native:` provenance, so all 20 are in scope. Reading `FXYP` from the retained final-cycle `MAS_OUT` and joining on `digest16 = sha256(pack_pattern)[:16]`:

**|F_xy_replay − F_xy_stored| = 0.000000 on 20/20.** Max over 20 = 0.000000, against the registered ≤ 0.002.

The `F_xya` axial companion likewise reproduces exactly on 20/20 (1.3708 – 1.4504). This is the substitute for the `post_verify_top_k = 5` re-verification r1 never ran, and it covers 20 rather than 5: **every `F_xy` number in the r1 results is replay-exact, and the 1.5322 → 1.65 margin needs no widening for replay noise.** The precedent's 40/40 at 0.0000 now extends to 60/60 across the two waves.

Group B is excluded from M2 by construction (`stored.f_xy` null) — its five values are first labels, reported in §5, not deltas.

---

## 4. M5 / M8 — head skill

### 4.1 M5 — the feed-121 pin bias replicates

| slice | n | bias (pred−meas) | MAE | sd | 95% CI |
|---|---:|---:|---:|---:|---|
| **pooled (all feed 121)** | 25 | **+3.420** | 3.420 | 0.485 | **[+3.23, +3.61]** |
| `minfxy_r1_top20` (delivery) | 20 | +3.54 | 3.54 | 0.47 | [+3.34, +3.74] |
| `frera_fxy_measure` (calibration) | 5 | +2.96 | 2.96 | 0.17 | [+2.81, +3.08] |
| *precedent, feed 121* | *13* | *+3.230* | *3.23* | — | *[+2.39, +4.07]* |
| *prior, in-support store labels* | *193* | *+0.16* | *1.21* | *1.72* | *[−0.09, +0.39]* |

**The CIs overlap** ([+3.23, +3.61] ⊂ [+2.39, +4.07]) and the point estimates differ by 0.19 GWd/tU on n = 13 → n = 25. The precedent's single most actionable number is **replicated at nearly double the n and one third the interval width**. `MAE = |bias|` again, exactly: **0 of 25 residuals is negative**, so the head is not noisy here — it is *uniformly conservative by ~3.4 GWd/tU* at this cell.

`pinbu_analyze.py` scores this **NEITHER** against the two registered hypotheses (H1 = +9 head bias, H2 = −1 ± 2 pool deficit) and trips the recalibration flag (|bias| 3.42 vs the 2.0 trigger). The registered fix stands: **fit `pinbu_physics` per library** (`lpopt/model/pinbu_physics.py::fit_pinbu_physics`) before the next champion; no champion `s1c..s1j` ships one, so the served pin column is still the raw head. The `frera_fxy_measure` calibration curve (`measured = 0.3862 × predicted + 37.917`, r = 0.218, n = 5) is **degenerate over a 0.23-wide predicted span, exactly as the prereg warned, and must be ignored.**

### 4.2 M8 — `f_xy` head level skill on the 20 labelled cores

Because measurement reproduced storage exactly, the manifest's pinned residuals stand verbatim: **bias −0.0075, median −0.0181, MAE 0.0238, sd 0.0309** on n = 20 — comfortably inside `s1j`'s `G2′ MAE < 0.0767` bar, on this cell. Level skill is real.

**Ranking skill is not.** Spearman ρ(predicted `F_xy`, measured `F_xy`) over the 20 = **−0.114** (p = 0.63), consistent with the r1 results §6.3 finding (joint-clean ρ +0.016, exploit-slot ρ −0.383). The head knows *where* this cell sits, not *which core inside it is better*. That is precisely the footing the M4 prediction was placed on, and §5 shows why it mattered.

---

## 5. M4 — `F_xy` of the `F_r`-record family → **PASS on the mark, MISS on the estimator**

**First measured `F_xy` labels for the `F_r`-era record cores of this cell.** All five exceed the r1 optimum's 1.5322, so the registered prediction is **not falsified** and the `min_fxy` objective retains the cell record.

| record | campaign | `F_r` | **meas `F_xy`** | `F_xya` | vs 1.5322 | `F_xy`/`F_r` | meas pin |
|---|---|---:|---:|---:|---:|---:|---:|
| `4d70ab6f75d4` | batchswap_enum_T6T4 | **1.4605** | **1.5402** | 1.4206 | **+0.0080** | 1.0546 | 63.669 |
| `0a83b6547650` | batchswap_enum_T6T4 | 1.4612 | **1.5471** | 1.4331 | +0.0149 | 1.0588 | 63.878 |
| `eaf914e573b2` | batchswap_enum_T6T4 | 1.4648 | **1.5480** | 1.4288 | +0.0158 | 1.0568 | 63.470 |
| `456199b370ad` | batchswap_enum_T6T4 | 1.4616 | **1.5498** | 1.4220 | +0.0176 | 1.0603 | 63.493 |
| `188c9a338d9f` | fpcamp_minfr_T6T4_r8 | 1.4749 | **1.5684** | 1.4295 | +0.0362 | 1.0634 | 63.665 |

### 5.1 Estimator scorecard — the model-free ratio won, decisively

| core | measured | s1j raw | mean-corr | med-corr | **ratio (1.0640 × F_r)** |
|---|---:|---:|---:|---:|---:|
| `4d70ab6f75d4` | **1.5402** | 1.5824 (+0.042) | 1.5900 (+0.050) | 1.6005 (+0.060) | **1.5540 (+0.014)** |
| `0a83b6547650` | **1.5471** | 1.6044 (+0.057) | 1.6120 (+0.065) | 1.6226 (+0.076) | **1.5547 (+0.008)** |
| `eaf914e573b2` | **1.5480** | 1.5807 (+0.033) | 1.5883 (+0.040) | 1.5988 (+0.051) | **1.5585 (+0.011)** |
| `456199b370ad` | **1.5498** | 1.5962 (+0.046) | 1.6038 (+0.054) | 1.6143 (+0.065) | **1.5551 (+0.005)** |
| `188c9a338d9f` | **1.5684** | 1.5997 (+0.031) | 1.6072 (+0.039) | 1.6178 (+0.049) | **1.5693 (+0.001)** |
| **MAE** | — | 0.042 | 0.050 | 0.060 | **0.008** |

Three findings, all registered as reportable and all consequential:

1. **The ratio estimator beat the learned head by 5–7×** (MAE 0.008 vs 0.042–0.060), and its 1.554 point estimate for `4d70ab6f` was within 0.014 of truth against the head's ≈1.59. The prereg named the ratio as the *secondary* estimator; it was the better one.
2. **Bias-correcting the head made it worse.** The group-A residual bias is **−0.0075** but the group-B residual bias is **+0.0420** — opposite sign. The correction fitted on the `F_xy`-era subset does not transfer to the `F_r`-era family, so the raw head was closer than either corrected variant. This is a concrete instance of the §4.2 ranking incompetence: the head systematically over-predicts `F_xy` for cores the `F_r` objective selected.
3. **The registered lower tail did not materialise.** The ratio estimator's worst case (1.0158 × 1.4605 = 1.484) was the prereg's honest escape hatch for a falsification; the measured ratios sit at **1.0546–1.0634**, a *tight* band well above the tail and just under the group-A median 1.0640. The 1.0158 outlier was `d06a56d6` alone, not a property of the low-`F_r` region.

### 5.2 The decision-grade consequence: r1's gain is half what it claimed

M4 is not falsified — the `F_r`-era search did **not** hold the cell's `F_xy` record. But it held **1.5402**, and the r1 results reported moving the incumbent **1.5491 → 1.5322**. That 1.5491 incumbent was the best *labelled* core, not the best core: `4d70ab6f` was unlabelled and better.

| framing | pre-r1 best `F_xy` | r1 best | gain |
|---|---:|---:|---:|
| as reported in r1 results (labelled subset) | 1.5491 | 1.5322 | **0.0169 (−1.09%)** |
| **corrected (full cell, this wave's labels)** | **1.5402** (`4d70ab6f`) | 1.5322 | **0.0080 (−0.52%)** |

**The marginal value of the `min_fxy` objective at `T6_T4`/f121 is positive but roughly halved.** Four of the five `F_r`-era cores (1.5402, 1.5471, 1.5480, 1.5498) would have ranked 3rd–4th inside the r1 top-20 — i.e. the `F_r` objective, optimising a different quantity entirely, produced cores competitive with all but the top two of a dedicated 95-row `F_xy` round. This directly gates the r2 cell choice registered in r1 results §11.4-5: **an r2 at this cell is buying ~0.008 of `F_xy` against a `min_fr` baseline, not ~0.017.**

Note also that `4d70ab6f` dominates `bf3a70b2` on every other axis — `F_r` 1.4605 vs 1.4857, `F_q` 1.807 vs 1.853, CBC 1302.7 vs 1337.4 — while giving up 0.0080 of `F_xy` and 4.1 EFPD of cycle length. Which of the two is the better *delivery* candidate is not an `F_xy` question, and both are now fully measured (§6).

### 5.3 M7 — the ratio at low `F_r`

Group B extends the measured (`F_xy`, `F_r`) pair set below `F_r` 1.4769 for the first time at this cell. Measured ratios **1.0546 – 1.0634, median 1.0588**, against the group-A stored span **1.0158 – 1.0697, median 1.0640** — inside the span, tighter than it, and every one at or below the group-A median. The ratio does **not** blow up as `F_r` falls; if anything it contracts.

**Caveat stands, as registered:** this cell still produced no converged core with `F_r` in [1.50, 1.65] below 1.5238, so the `S1` wall-location interval [1.04, 1.08] is **again not scorable in the region it was registered for**. All five group-B ratios do fall inside [1.04, 1.08], which is consistent with but not a test of `S1`.

---

## 6. DELIVERY VERDICT — the r1 optimum `bf3a70b2` is **DELIVERABLE**

`is_deliverable` (`lpopt/search/campaign.py:567`) requires every gated licensing axis **measured** and inside its limit. For `bf3a70b20e50…`:

| axis | limit | measured | margin | verdict |
|---|---:|---:|---:|:-:|
| `f_xy` | ≤ 1.65 | **1.5322** | 0.1178 | ✔ |
| `f_r` | ≤ 1.55 | **1.4857** | 0.0643 | ✔ |
| `cbc_max` | ≤ 1600 | **1337.38** | 262.6 | ✔ |
| `f_q` | ≤ 2.41 | **1.8530** | 0.557 | ✔ |
| \|AO\| | ≤ 0.30 | **0.0244** | 0.2756 | ✔ |
| `max_pin_burnup` | ≤ 80.0 | **63.760** | 16.24 | ✔ |
| `cyclen` | (band) | 622.101 | — | ✔ |

**`unknown_axes = ()` · `is_deliverable = True`.** The round's optimum has a DELIVERY verdict for the first time, and it is a pass on every axis with the *tightest* margin being `F_r` at 0.064 — not pin, and not `F_xy`.

**Deliverable-grade population of this wave:**

| set | n | deliverable | binding axis |
|---|---:|---:|---|
| `minfxy_r1_top20` | 20 | **20/20 (100%)** | `F_r` (worst 1.5238, margin 0.026) |
| `frera_fxy_measure` | 5 | **5/5 (100%)** | `F_r` (worst 1.4749, margin 0.075) |
| **wave total** | 25 | **25/25** | — |

No core in either set is limited by pin burnup, `F_q`, CBC or AO; the whole population sits 4–7% under `F_xy` 1.65 and the only axis with a margin under 0.1 is `F_r`.

---

## 7. Store merge

Backup taken before any write: `E:/lpopt_data/5_RL/backups/records.parquet.bak_pre_pinbu_minfxy_r1_20260830` (sha256 `73701E33…0C85F`, byte-identical to the pre-merge store).

**Step 1 — `pinbu_wave.py patch` (`max_pin_burnup`, tag `pinbu_minfxy_r1_20260830`)**

```
[patch] 25 result row(s): 25 accepted, 0 refused
[patch] would write max_pin_burnup on 25 row(s) (25 also carry max_assembly_burnup)
[patch] wrote 25 measured pin value(s); store now 76693 row(s)
```

Dry-run and real run agree exactly. **25 accepted / 0 refused** — no provenance refusal, unlike the precedent's 8.

**Step 2 — `python -m lpopt.tools.backfill_fxy apply` (`f_xy` / `f_xya`)**

```
final & sane 25 -> 25 digest(s) | dup 0 | no store row 0 | ambiguous 0
cycle mismatch 0 | F_r<=F_xy<=F_q violations 0 | already filled 20
populated 5   (batchswap_enum_T6T4 4, fpcamp_minfr_T6T4_r8 1)
```

Exactly as registered: 5 populated, 20 already filled. Every one of the 25 passed the `F_r ≤ F_xy ≤ F_q` sanity gate and the cycle-evidence check.

**Store after both merges**

| field | before | after |
|---|---:|---:|
| rows | 76,693 | **76,693** |
| `f_xy` non-null | 7,662 | **7,667** (+5) |
| `max_pin_burnup` non-null | 40,845 | **40,870** (+25) |
| rows with **both** measured | 81 | **106** (+25) |
| **joint-clean + deliverable (6 axes)** | **0** | **25** |
| bytes | 22,780,281 | **22,780,663** |
| sha256 | `73701E33F07291E17609BA30D025E2A5B7A423FEB69F08D23DE4EC23EBE0C85F` | **`F7430821400523D1A30E18EF5DB3FE21FF302A81EAA72433F04AA9B51694B5EB`** |

**Store-wide, the deliverable set is exactly this wave's 25 rows** — all `T6_T4` / feed 121 / paramA, `F_xy` 1.5322–1.5890, `F_r` 1.4605–1.5238, pin 62.451–63.878. Before this wave the store held **zero** rows with every licensing axis measured and inside limit: the 81 pre-existing dual-measured rows are all `F_xy`-era cores at 1.78–2.39, far outside the 1.65 gate. **This wave is the origin of the store's deliverable population.**

As prereg §8 flagged, the store sha256 has moved again; any launcher `$wantSha` and the r1 results §9.1 row must be re-stamped to `F7430821…B5EB`.

---

## 8. What this wave settles, and what it opens

**Settled.**
1. The r1 optimum is deliverable, with 16.2 GWd/tU of pin margin. The `enable_pin_burnup` blind spot (`verify.py:851`) is closed for this round's frontier by the `pinbu_wave.py` path.
2. r1's `F_xy` labels carry **zero** replay noise (20/20 exact), so no r1 margin needs widening.
3. Pin burnup is **not** a live constraint at feed 121; the `+3.4` conservative head bias is now replicated at n = 25 with a tight CI.
4. Plan-time `native:`-only provenance selection eliminates the precedent's promoted-cache drift entirely (25/25 vs 32/40).

**Opened.**
1. **r1's gain is 0.0080, not 0.0169** (§5.2). The r2 cell-choice decision registered in r1 results §11.4-5 should be re-taken on the corrected number. A cell where `min_fr` already lands within 0.008 of a dedicated `min_fxy` round is a weak place to spend an r2.
2. **The `f_xy` head's bias flips sign between the `F_xy`-era and `F_r`-era families** (−0.0075 vs +0.0420). Any future bias correction must be fitted per-family or not at all, and the model-free ratio estimator should be carried as a first-class baseline — it beat the head 5×.
3. **`pinbu_physics` is still unfitted** for every champion through `s1j`. With the feed-121 bias now pinned at +3.42 (CI [+3.23, +3.61]) on n = 25 and the feed-109 deficit at −2.65 from the precedent, there is enough labelled support to fit paramA. Recommended before the next champion.
4. **`4d70ab6f` vs `bf3a70b2` is now a real delivery choice** (§5.2): the former wins `F_r`, `F_q` and CBC, the latter wins `F_xy` by 0.008 and cycle length by 4.1 EFPD. Both are fully measured. Which one ships is a licensing-priority question, not a search question.
5. The `frera_fxy_measure` calibration curve is degenerate (predicted span 0.23) and must not be used. A real pin calibration set needs a *wide* predicted span, which this cell cannot supply.

---

*Generated 2026-08-30. Sources: `runs/pinbu_wave_minfxy_r1/pinbu_wave_results.jsonl` (25 rows, rc = 0), `runs/pinbu_wave_minfxy_r1/pinbu_wave_minfxy_r1_out.log` (wave wall 1,262 s), `data/reports/fxy_backfill_199_pinbu_wave_minfxy_r1_20260830.csv` (25 sane, joined on `digest16 = sha256(pack_pattern)[:16]`), `data/reports/pinbu_wave_minfxy_r1_manifest.json`, `data/store/records.parquet` post-merge.*
