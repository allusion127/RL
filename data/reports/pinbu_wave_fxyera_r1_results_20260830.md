# Phase-2 pin-burnup + F_xy determinism wave — RESULTS (F_xy-era r1, 2026-08-30)

**Pre-registration** `data/reports/pinbu_wave_fxyera_r1_prereg_20260830.md` (binding)
**Manifest** `data/reports/pinbu_wave_fxyera_r1_manifest.json` · **Run dir** `runs/pinbu_wave_fxyera_r1`
**Deck** `pinbu_wave_fxyera_r1_199.inp` · **Box** HOST_199 · **Harness** `pinbu_wave.py run` / `patch`
**Scored against the prereg as written.** No deck, harness or analysis code was changed for this readout.

---

## 0. Marks

| mark | registered prediction | measured | verdict |
|---|---|---|---|
| **M1** pin ≤ 80 on `fxywall_top20` | **≥ 80% of 20** | **14/20 = 70.0%** | **FALSIFIED** |
| **M2** `\|ΔF_xy\|` on determinism-ok replays | **≤ 0.002, max over 40** | **0.0000 on 40/40** | **PASS** (exact) |
| M3 chain integrity | 40/40 converged, 40/40 `provenance_ok` | 40/40 converged, **32/40** `provenance_ok` | **PARTIAL** |
| M4 pin-head skill, `F_xy`-selected | (no point prediction; 2026-08-20 gave bias −0.32 / MAE 1.02) | bias **+1.215**, MAE **2.193**, n=40 | reported |
| M5 scalar determinism | reproduce 2026-08-20 (`f_r` 0.0000, `cyclen` 0.008, `cbc` 0.01, `f_q` 0.0001) | `f_r` **0.0000**, `cyclen` 0.006, `cbc` 0.02, `f_q` 0.0001 | reported — matches |
| M6 `F_xy`/`F_r` ratio | stored median 1.0758, span 1.0240–1.1220 replicated | **identical to stored on 40/40** | reported — not scorable for `S1` |

Run wall **2,104 s** (0.58 h) against the 1.6 h registered cap and the ~0.6 h expectation — **inside budget**.
Per-chain wall median 349.6 s, max 638.5 s (precedent: median 359 s, max 630 s).

---

## 1. Run integrity (M3)

40/40 chains converged, 40/40 `determinism_ok`, **32/40 `provenance_ok`**. The registered
40/40 provenance expectation is missed.

**The drift ran in the opposite direction from the registered risk.** §7 of the prereg
registered the 2026-08-20 failure mode: a *pruned* promoted cache makes a `promoted:` target
fall back to `pair_*`. What happened is the reverse — every one of the 8 targets whose **store
row records a `pair_ecore:`/`pair_feed:` restart resolved a `promoted:` restart on 199**:

| record | group | campaign | planned (stored) | resolved | measured pin | stored `F_xy` |
|---|---|---|---|---|---:|---:|
| `43d81bf75374` | fxywall_top20 | fxy_exp_HGD569_f125 | `pair_ecore:…_11_0705.02` | `promoted:…_21_0729.46` | 75.001 | 1.8773 |
| `f0da920dcabd` | fxywall_top20 | fxy_exp_HGD569_f109 | `pair_feed:…_21_0729.46` | `promoted:…_21_0654.26` | 97.220 | 1.9118 |
| `bec9339d8845` | fxywall_top20 | fxy_exp_HGD569_f125 | `pair_ecore:…_11_0705.02` | `promoted:…_21_0729.46` | 79.388 | 1.9168 |
| `b58578254a89` | fxywall_top20 | fxy_exp_HGD569_f125 | `pair_ecore:…_11_0705.02` | `promoted:…_21_0729.46` | 78.654 | 1.9284 |
| `6db46706030c` | fxywall_top20 | fxy_exp_HGD569_f109 | `pair_feed:…_21_0729.46` | `promoted:…_21_0654.26` | 94.299 | 1.9367 |
| `99b0d0b2bdee` | replicate_span20 | fxy_exp_HGD569_f125 | `pair_ecore:…_11_0705.02` | `promoted:…_21_0729.46` | 82.011 | 2.0775 |
| `bf9161ac382a` | replicate_span20 | fxy_exp_HGD569_f109 | `pair_feed:…_21_0729.46` | `promoted:…_21_0654.26` | 87.341 | 2.1091 |
| `c60a9be5d1a1` | replicate_span20 | fxy_ood_e614_f129 | `pair_feed:…_11_0767.69` | `promoted:…_19_0796.67` | 92.396 | 2.3876 |

Registered mix `promoted` 18 / `native` 14 / `pair_ecore` 4 / `pair_feed` 4 → resolved mix
`promoted` **26** / `native` 14. **All 18 `promoted:` and all 14 `native:` targets resolved
correctly; the drift is exactly the 8 `pair_*` targets, 100% of them.** The cause is the
mirror of the registered one: r1 itself *populated* those promoted cells after the store rows
were written, so the resolver now prefers a promoted restart where r1 had none. Prereg §7's
mitigation 1 (launcher refuses if `data\produce\promoted` is missing) cannot see this — the
cache was present, it had *grown*. **A launcher gate on promoted-cache cell count, not mere
presence, is the fix; it needs its own pre-registration.**

**Side finding, recorded as a finding and not a result** — the same one 2026-08-20 §7 recorded,
now strengthened. All 8 drifted chains reproduced their stored `f_r` to **0.0000**, `cyclen`,
`cbc_max`, `f_q`, `ao_abs` inside tolerance, **and `F_xy` to 0.0000** from a *different* seed
restart. The equilibrium in these cells is restart-independent, and this wave extends that
statement from the scalar FOMs to `F_xy` itself. That would justify a future amendment
relaxing the provenance control. It does not license one here: the 8 stay refused.

---

## 2. M1 — pin axial peak on the `F_xy` frontier → **FALSIFIED**

**14 of 20 measure ≤ 80.0 GWd/tU = 70.0%**, against the registered **≥ 80%**. The ≤ 78.0
acquisition-gate rate is **11/20 = 55%** (s1i predicted 9/20 = 45%). Restricting to the 15
chains that also hold provenance does not rescue the mark: **11/15 = 73.3% ≤ 80**, 10/15 =
66.7% ≤ 78.

Measured span **64.479 – 97.220**, median 77.759.

| record | campaign | feed | stored `F_xy` | pred pin | **meas pin** | pred−meas | prov_ok | ≤80 |
|---|---|---:|---:|---:|---:|---:|---|---|
| `7627c10841fe` | fxy_exp_S3T1S5_f125 | 125 | 1.7836 | 76.77 | **75.000** | +1.77 | ✔ | ✔ |
| `531d5a436d9e` | fxy_exp_HGD569_f125 | 125 | 1.8405 | 77.66 | **77.583** | +0.08 | ✔ | ✔ |
| `3f77a1f570ea` | fxy_bnd_T3T4_f121 | 121 | 1.8454 | 70.09 | **65.655** | +4.44 | ✔ | ✔ |
| `e5cd06b2b57f` | fxy_exp_HGD569_f109 | 109 | 1.8607 | 86.97 | **91.061** | −4.09 | ✔ | ✘ |
| `43d81bf75374` | fxy_exp_HGD569_f125 | 125 | 1.8773 | 79.01 | **75.001** | +4.01 | ✘ | ✔ |
| `9ebd5831d8c2` | fxy_ood_T6T4_f109 | 109 | 1.8888 | 79.28 | **80.766** | −1.49 | ✔ | ✘ |
| `7c38492e122a` | fxy_bnd_T3T4_f121 | 121 | 1.8952 | 69.25 | **64.479** | +4.77 | ✔ | ✔ |
| `4f6e002473b4` | fxy_bnd_T3T4_f121 | 121 | 1.9007 | 72.26 | **67.807** | +4.45 | ✔ | ✔ |
| `6207fd6e8216` | fxy_exp_HGD569_f125 | 125 | 1.9057 | 78.14 | **78.536** | −0.39 | ✔ | ✔ |
| `048fedfa1a14` | fxy_exp_T6T4_f121 | 121 | 1.9064 | 71.11 | **66.097** | +5.01 | ✔ | ✔ |
| `d0c9a51a1eee` | fxy_exp_S3T1S5_f125 | 125 | 1.9090 | 77.24 | **76.010** | +1.23 | ✔ | ✔ |
| `f0da920dcabd` | fxy_exp_HGD569_f109 | 109 | 1.9118 | 94.73 | **97.220** | −2.49 | ✘ | ✘ |
| `bec9339d8845` | fxy_exp_HGD569_f125 | 125 | 1.9168 | 78.97 | **79.388** | −0.42 | ✘ | ✔ |
| `8449def1d565` | fxy_exp_S3T1S5_f125 | 125 | 1.9185 | 78.25 | **77.935** | +0.32 | ✔ | ✔ |
| `b58578254a89` | fxy_exp_HGD569_f125 | 125 | 1.9284 | 79.10 | **78.654** | +0.45 | ✘ | ✔ |
| `30f4a07b8343` | fxy_exp_S3T1S5_f125 | 125 | 1.9334 | 78.98 | **76.407** | +2.57 | ✔ | ✔ |
| `467cb54ab299` | fxy_exp_T6T4_f121 | 121 | 1.9346 | 72.35 | **67.134** | +5.22 | ✔ | ✔ |
| `6db46706030c` | fxy_exp_HGD569_f109 | 109 | 1.9367 | 94.15 | **94.299** | −0.15 | ✘ | ✘ |
| `a0d1e59917d8` | fxy_bnd_S3T1S5_f125 | 125 | 1.9536 | 79.76 | **80.651** | −0.89 | ✔ | ✘ |
| `763ac845601b` | fxy_ood_T6T4_f109 | 109 | 1.9544 | 75.73 | **80.737** | −5.01 | ✔ | ✘ |

### 2.1 The failure is a feed effect, not an `F_xy`-frontier effect

The falsifier's registered reading — "the `F_xy`-selected frontier is pin-limited" — is true as
stated but **too coarse to act on**. Split by feed it resolves completely:

| feed | n | ≤ 80 | ≤ 78 | measured span | head bias (pred−meas) |
|---|---:|---:|---:|---|---:|
| 109 | 5 | **0** | **0** | 80.737 – 97.220 | −2.65 |
| 121 | 5 | **5** | **5** | 64.479 – 67.807 | +4.78 |
| 125 | 10 | **9** | 6 | 75.000 – 80.651 | +0.87 |

**Every one of the six ≤ 80 failures is a feed-109 core (5/5) or the single f125 core sitting
0.65 above the limit (`a0d1e59917d8`, 80.651).** Feed 121 clears the limit by 12–15 GWd/tU with
no near misses; feed 125 clears it 9/10. Across all 40 chains the pattern holds and hardens:
**feed 109 is 0/7 ≤ 80**, feed 129 0/1, feed 117 1/3, while feed 121 is 10/13 and feed 125
11/16.

The correlation between stored `F_xy` and measured pin is weak on both sets — Pearson r
**0.176** on the 20 wall cores, **0.281** over all 40. **Low `F_xy` does not buy low pin
burnup, and it does not cost it either; the two are close to orthogonal on this support, and
feed dominates both.**

**Decision-grade consequence, as registered:** pin is a live constraint on `F_xy` search
**at feed 109 and below-121 feeds generally**, and an advisory column at feed 121/125. A
`min_fxy` campaign that draws its cores from feed 109 must carry a measured pin gate; one at
feed 121 need not.

---

## 3. M2 — `F_xy` determinism → **PASS, exactly**

All 40 replays are `determinism_ok`, so all 40 enter the mark. Joining the re-measured
`F_xy` (`fxy_backfill_199_pinbu_wave_fxyera_r1_20260830.csv`, 40 rows, **40 `sane=1`**, joined
to the store on `sha256(pattern)[:16]` — the same `Pattern.digest` key
`lpopt/tools/backfill_fxy.py apply` uses) against the stored `f_xy`:

**max |ΔF_xy| = 0.000000 over 40/40.** Mean |Δ| 0.000000; **zero** chains show any
difference at all. Same for the axial companion: **max |ΔF_xya| = 0.000000, 40/40.**

Registered bound 0.002; measured 0.0000. The prediction holds with the full 0.002 to spare,
and holds on the 8 provenance-drifted chains too (max |Δ| 0.0000 on both the 32 and the 8).

*Resolution caveat, stated so the zero is not over-read:* both the scanner and the stored
label carry `F_xy` at 4 decimal places, so "0.000000" means the two agree to the last reported
digit — i.e. |ΔF_xy| < 5×10⁻⁵ — not that the underlying float is bit-identical. That is an
order of magnitude tighter than the mark needed.

**Consequence:** the 540 r1 `f_xy` labels carry **no measurable replay noise**. `F_xy` margins
quoted against the 1.65 licensing limit need **no** widening for evaluation noise. The margin
that matters is entirely physical distance, and on r1's support that distance is 2.6% (the
whole-store minimum 1.6926, `5cecf2d73b1b`).

`backfill_fxy.py apply` was **not** run: these 40 are replays of rows already carrying `f_xy`,
so `apply` would report "already filled" and write nothing. The CSV was used for the M2
comparison only. Store `f_xy` non-null is unchanged at 6,776.

---

## 4. M5 — scalar determinism, restated

Max |Δ| (measured − stored) over the 40 chains, against `TOL_F_R`/`TOL_CYCLEN`/`TOL_CBC` =
0.002 / 0.5 / 2.0:

| quantity | max \|Δ\| | chains with Δ≠0 | 2026-08-20 (44 chains) |
|---|---:|---:|---:|
| `f_r` | **0.000000** | 0/40 | 0.0000 |
| `f_q` | 0.000100 | 2/40 | 0.0001 |
| `cbc_max` | 0.020000 | 3/40 | 0.01 |
| `cyclen` | 0.006000 | 6/40 | 0.008 |
| `ao_abs` | 0.000100 | 1/40 | — |
| `max_assembly_burnup` | 0.003000 | 1/40 | ≤0.001 |
| **`F_xy`** | **0.000000** | **0/40** | not measurable |

MASTER's determinism on this fleet is reproduced at the 2026-08-20 level: `f_r` exact on
40/40, everything else two to three orders inside its tolerance. `max_assembly_burnup` agreed
so closely that the patch left the stored column **byte-identical** (§6).

---

## 5. M4 — pin-head skill on an `F_xy`-selected set

First test of the s1i pin head on cores selected by `F_xy` rather than `F_r`.
`pin_error_pred_minus_meas`, positive = head over-predicts (conservative):

| slice | n | bias | MAE | sd | 95% CI on bias |
|---|---:|---:|---:|---:|---|
| **POOLED** | 40 | **+1.215** | **2.193** | 2.481 | [+0.45, +1.98] |
| `fxywall_top20` (delivery) | 20 | +0.969 | 2.463 | 3.034 | [−0.36, +2.30] |
| `replicate_span20` (calibration) | 20 | +1.461 | 1.924 | 1.817 | [+0.67, +2.26] |
| feed 109 | 7 | **−1.416** | 2.362 | 2.779 | [−3.48, +0.64] |
| feed 117 | 3 | +1.153 | 1.153 | 0.458 | [+0.64, +1.67] |
| feed 121 | 13 | **+3.230** | 3.230 | 1.549 | [+2.39, +4.07] |
| feed 125 | 16 | +1.046 | 1.380 | 1.613 | [+0.26, +1.84] |
| feed 129 | 1 | −3.664 | 3.664 | — | — |
| `provenance_ok` only | 32 | +1.519 | 2.264 | 2.398 | [+0.69, +2.35] |
| provenance-drifted | 8 | −0.000 | 1.909 | 2.593 | [−1.80, +1.80] |

**Against 2026-08-20 (`F_r`-selected: bias −0.32, MAE 1.02):** the pooled bias has flipped
sign and roughly quadrupled in magnitude, and MAE has **doubled**. The pooled 95% CI
[+0.45, +1.98] now **excludes zero** — on 2026-08-20 it did not ([−0.74, +0.08]). The head is
measurably biased on `F_xy`-selected cores in a way it was not on `F_r`-selected ones. This is
the `E5` caveat of 2026-08-20 §3 playing out exactly as it warned: that wave's slices were
largely in-sample, and this set is not.

**Recalibration trigger: pooled |bias| 1.215 against the 2.0 threshold → NOT triggered.**
The serve path stays as it is. But the pooled number hides the operative split, and the
per-feed rows are the ones to carry forward:

* **feed 121, bias +3.23 (n=13, CI [+2.39, +4.07])** — the head over-predicts by more than the
  entire 2.0 GWd/tU deck margin, on every one of 13 cores (MAE = bias, i.e. **no negative
  residual anywhere in the slice**). `minfr_pin_bu_limit = 78` is therefore rejecting feed-121
  cores that measure in the mid-60s. This is a *throughput* loss, not a safety issue, and it is
  large.
* **feed 109, bias −1.42 (n=7)** — the head **under**-predicts, the direction the 2.0 margin
  exists to absorb, and the worst single residual is **−5.01** (`763ac845601b`: predicted
  75.73, measured **80.737**). That one core **passes the 78 acquisition gate and violates the
  80 licensing limit** — 1 of 20 on the wall set, 1 of 40 overall. Three cores pass a
  predicted-≤80 screen and measure >80 (`9ebd5831d8c2`, `a0d1e59917d8`, `763ac845601b`).

**The 2.0 margin does not cover feed 109.** It is exceeded by 3.0 GWd/tU on the worst core here,
consistent with the `E4`/`E5` finding that new-basin under-prediction reaches −6.

### 5.1 Calibration curve (`replicate_span20`)

`measured = 1.0474 × predicted − 5.373` (n=20, r = **0.974**, predicted span 69.4–94.2 →
measured 67.3–93.3). The 2026-08-20 fit was `1.0320 × predicted − 1.953` (n=21, r=0.933) — the
same slope to within 1.5%, shifted down 3.4 GWd/tU. **Not claimed outside the fitted predicted
span, and — per `E5b` — not to be extrapolated into the delivery region.**

---

## 6. Store merge

```
python pinbu_wave.py patch --results runs/pinbu_wave_fxyera_r1/pinbu_wave_results.jsonl --dry-run
python pinbu_wave.py patch --results runs/pinbu_wave_fxyera_r1/pinbu_wave_results.jsonl --tag pinbu_fxyera_r1_20260830
```

Dry-run and real patch agreed exactly: **40 result rows → 32 accepted, 8 refused**, all 8
refused on `restart provenance changed` (§1). No refusal on `status`, on a missing PPI value,
or on `determinism_ok`. No `--allow-overwrite` was needed or passed — none of the 32 rows
already carried a `max_pin_burnup`.

* backup `data/store/records.parquet.bak_pre_pinbu_fxyera_r1_20260830`, written by the tool
  **before** the store write, then **moved to `E:/lpopt_data/5_RL/backups/`** per standing
  policy. Its sha256 is `F38666E9…FECA6EA` — **identical to the store hash the prereg pinned in
  §8**, so the determinism reference the manifest was built against is preserved byte-for-byte.
* **32** rows patched in place by `record_id`; **75,793** rows, row order preserved
  (`record_id` sequence identical to the backup on all 75,793).
* **Exactly one column changed: `max_pin_burnup`, 32 cells.** Verified by column-wise
  comparison of the post-patch store against the backup — every other column, including
  `max_assembly_burnup`, `f_xy`, `f_xya`, `maps_key`, `node_peak`, `map_cov`, is unchanged.
  `max_assembly_burnup` was re-measured on all 32 and agreed to **0.000 on 32/32**, so the
  tool's write was a no-op on that column.
* `max_rod_avg_burnup` was **not** written — `cmd_patch` writes only `max_pin_burnup` and
  `max_assembly_burnup`, and `WaveFom` carries no rod-average field. The column stays at 5
  non-null rows (its `pinbu_rodavg_20260820` provenance).

| column | before | after |
|---|---:|---:|
| rows | 75,793 | 75,793 |
| `max_pin_burnup` non-null | 40,813 | **40,845** (+32) |
| `f_xy` non-null | 6,776 | 6,776 (unchanged) |
| `f_xya` non-null | 6,776 | 6,776 (unchanged) |
| `max_assembly_burnup` non-null | 66,546 | 66,546 (unchanged) |
| `max_rod_avg_burnup` non-null | 5 | 5 (unchanged) |

These are the programme's **first measured pin-burnup labels on `F_xy`-selected cores**.
Dataset-P fit pool for `pinbu_physics` / the pin head, after the merge (Dataset A's 38,854 pin
labels remain censored by `dataset_torch.censor_dataset_a_pin_labels`, so this pool is the
head's entire teacher):

| feed | 101 | 109 | 113 | 117 | **121** | 125 | 133 | 141 | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 129 | 462 | 22 | 749 | **0** | 310 | 150 | 137 | 1,959 |
| **after** | 129 | **466** | 22 | **752** | **13** | **322** | 150 | 137 | **1,991** |

**Feed 121 goes 0 → 13: the first measured pin labels at that feed anywhere in the P pool** —
and, per §5, the cell where the head is most badly miscalibrated (+3.23). Feed 129 gains
**nothing**: its single chain (`c60a9be5d1a1`) is one of the 8 provenance refusals, so the
feed-129 pin measurement of 92.396 exists in the results JSONL and in §5's statistics but is
**not** in the store.

---

## 7. M6 — `F_xy`/`F_r` on measured pairs

Because `ΔF_xy = 0.0000` and `Δf_r = 0.0000` on 40/40, the measured ratios are **identical to
the stored ratios to four decimals** — the replication is exact rather than merely consistent:

| set | n | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|
| stored (manifest, 40 targets) | 40 | 1.0240 | 1.0571 | **1.0758** | 1.0916 | 1.1220 |
| **measured (this wave)** | 40 | 1.0240 | 1.0571 | **1.0758** | 1.0916 | 1.1220 |

Median 1.0758 is inside r1 secondary `S1`'s registered interval [1.04, 1.08], in its upper
half — as the prereg pinned in advance.

**`S1` remains unscorable in the region it was registered for.** The measured `F_r` span over
these 40 is **1.6529 – 2.2268**; **0 cores** fall in `F_r ∈ [1.50, 1.65]`. The minimum, 1.6529,
belongs to `5cecf2d73b1b` — the global minimum-`F_xy` core — and it misses the region's ceiling
by 0.003. These ratios are an extrapolation toward the registered region, not a test of it.
The wave adds precision to the extrapolation (the ratios are now confirmed measurements, not
single-evaluation labels) and nothing to its reach.

---

## 8. Disposition — what this means for `min_fxy` delivery gating

The prereg made this the readout's purpose, so it is answered directly.

**Of the 20 low-`F_xy` cores, 11 are deliverable-grade on pin.** That is the count holding
*both* controls: `provenance_ok` (so the pin value belongs to the stored evaluation) **and**
measured pin ≤ 80.0. Against the 78.0 acquisition gate the count is **10**.

| filter | n of 20 |
|---|---:|
| measured pin ≤ 80 | 14 |
| measured pin ≤ 78 | 11 |
| `provenance_ok` | 15 |
| **`provenance_ok` AND pin ≤ 80 — deliverable-grade on pin** | **11** |
| `provenance_ok` AND pin ≤ 78 | 10 |
| stored `F_xy` ≤ 1.65 (licensing limit) | **0** |

**The binding constraint is `F_xy`, not pin — and that has not changed.** All 20 of these
cores, the lowest-`F_xy` joint-clean cores r1 produced, sit at `F_xy` 1.7836–1.9544, i.e.
**8–18% above** the 1.65 limit; the whole-store minimum over 540 labels, 1.6926, is 2.6% above
it and is not in this set (it is in `replicate_span20`, and it measures pin **76.867**, clean).
**Zero cores are deliverable on `F_xy` today.** Pin does not gate delivery because nothing has
reached the gate.

What pin *does* change is the **search**, and here the result is actionable:

1. **Pin is now a live constraint on `F_xy` search at feed 109.** 0/7 feed-109 cores in this
   wave measure ≤ 80, and the head under-predicts there by up to 5.0 — more than the deck's
   2.0 margin. A `min_fxy` campaign at feed 109 that trusts predicted pin will deliver cores
   that violate the limit. **Either gate feed-109 `min_fxy` candidates on measured pin, or do
   not spend `min_fxy` budget at feed 109.**
2. **Pin is not a constraint at feed 121, and the deck is throwing away headroom there.**
   13/13 feed-121 cores measure 64–73 while the head predicts 69–93, a uniform +3.23
   over-prediction with no negative residual anywhere in the slice. The 78 gate is rejecting
   feed-121 cores with 12+ GWd/tU of real margin. If `min_fxy` search is feed-121-heavy, the
   pin gate is costing candidates for nothing.
3. **`F_xy` labels need no noise allowance.** M2's exact zero means the 2.6% gap to the limit
   is entirely physical. Closing it is a search problem, and the `F_xy` numbers used to steer
   that search can be trusted at face value.
4. **The 8 refused chains are not lost information.** Their pin values are real measurements
   of a real (differently-seeded) equilibrium that reproduced every stored scalar *and* `F_xy`
   exactly. They are refused from the store because the pre-registered rule refuses them, and
   the rule is right until an amendment says otherwise. **Three of the five refused wall cores
   measure ≤ 80** (`43d81bf75374` 75.001, `b58578254a89` 78.654, `bec9339d8845` 79.388) — they
   are exactly the gap between the 14 wall cores that measure ≤ 80 and the 11 counted as
   deliverable-grade above.

**Recommended next steps, none launched, each needing its own pre-registration:**

* A launcher gate on promoted-cache **cell count** (not presence), which would have caught §1's
  drift before 0.58 h was spent.
* An amendment relaxing the provenance control, justified by two independent waves
  (2026-08-20 §7 and §1 here) showing restart-independent equilibria — now including `F_xy`.
* A feed-stratified pin margin in the decks, replacing the single `minfr_pin_bu_limit = 78`:
  the measured bias is +3.23 at feed 121 and −1.42 at feed 109, and one number cannot serve
  both.

---

## 9. Provenance and hashes

| artifact | bytes | sha256 |
|---|---:|---|
| `pinbu_wave_fxyera_r1_199.inp` | 7,345 | `9DF97D687652F7BA5FCD17234629AAF82134BED57E8A6CB790EB621E7620FD4B` |
| `data/reports/pinbu_wave_fxyera_r1_manifest.json` | 90,857 | `53D139C2BA7A42207EF0BED9DF26B558A62B76750AD9E88797434434E6E00D91` |
| `pinbu_wave.py` | 36,222 | `5B3688CFAD684E9E837910F8842F68A2F2C21F931052DD52DE98262BA3581047` |
| `runs/pinbu_wave_fxyera_r1/pinbu_wave_results.jsonl` | 60,013 | `D14A190C9CDEA2A655701CCD28B604936D5E3D2AC294B7572A95FAE8EFE495B3` |
| `runs/pinbu_wave_fxyera_r1/pinbu_wave_fxyera_r1_out.log` | 4,369 | `A6DE2B4C22C67D9AC53550D9255D2040938FD9E5B1DE0004DDA1348DE7C305C3` |
| `data/reports/fxy_backfill_199_pinbu_wave_fxyera_r1_20260830.csv` | 8,264 | `167E3BA10EE1084CBED27E6F83B33C398F33C3FDC0E8CF78DD29B8181637D325` |
| `data/store/records.parquet` **(pre-patch)** | 22,538,411 | `F38666E9F1508D35D33E0C22F583C5479C6F09CAC748201B494B47C8CFECA6EA` |
| `data/store/records.parquet` **(post-patch)** | 22,538,804 | `72516916F5D59A738BA95CE2A7D56F0F2E9F514E61DD654BE4BE6127D175CE5D` |
| `E:/lpopt_data/5_RL/backups/records.parquet.bak_pre_pinbu_fxyera_r1_20260830` | 22,538,411 | `F38666E9F1508D35D33E0C22F583C5479C6F09CAC748201B494B47C8CFECA6EA` |

The deck, manifest and harness hashes are **identical to the values pinned in prereg §8**, and
the pre-patch store hash is **identical to the §8 store gate** — the wave was scored against
exactly the artifacts it was registered against, with the store untouched between
pre-registration and readout.

*Written 2026-08-30 after the automated readout. Marks scored strictly per the
pre-registration; §1's provenance miss is reported, not absorbed.*
