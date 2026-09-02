# Pre-registration — phase-2 pin-burnup + determinism wave (min_fxy T6_T4/f121 r1, box HOST_199)

**Date** 2026-08-30 · **Deck** `pinbu_wave_minfxy_r1_199.inp` · **Run dir** `runs/pinbu_wave_minfxy_r1`
**Harness** `pinbu_wave.py run` (NOT `lpopt optimize`, NOT `lpopt produce`)
**Manifest** `data/reports/pinbu_wave_minfxy_r1_manifest.json` (`schema: pinbu_wave_prereg_v1`, 25 targets)
**Registered by** `data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md` §5.4 (phase-2) and
`data/reports/minfxy_T6T4_f121_r1_results_20260830.md` §11.3 (the recommendation, incl. the
`frera_fxy_measure` addition it says needs its own pre-registration — this document)
**Precedent** `data/reports/pinbu_wave_fxyera_r1_prereg_20260830.md` / `…_results_20260830.md`
**Store at write time** `data/store/records.parquet`, **75,893 rows**, sha256 pinned in §8
**Status** WRITTEN BEFORE ANY MASTER CALL. Hashes pinned in §8. **Nothing launched.**

---

## 0. One-paragraph statement

The `min_fxy` round r1 on `T6_T4/f121` has drained, merged and been read out. It moved the cell
incumbent `1.5491 → 1.5322` and it left three things **structurally** unanswerable inside its
own run (results §11.2, D1): the optimum has **no measured pin burnup** — 0 of 95 rows, because
`verify.py:851` hard-codes `enable_pin_burnup=False` on the optimize path — so it has no
**DELIVERY** verdict; `post_verify_top_k = 5` never fired, so nothing re-verified the round's
own `F_xy` numbers; and the `F_r`-era record cores of the same cell carry **no `f_xy` label at
all**, so "`min_fxy` picks a better core than `min_fr`" is a claim about a labelled subset, not
about the cell. `pinbu_wave.py` closes all three in one 25-chain pass, on the exact pattern of
`pinbu_wave_fxyera_r1`, with **no code change and no new strata**. It also carries one deliberate
correction to that precedent: its M3 provenance mark failed 8/40 because the produce promoted
cache had *grown* and re-resolved eight `pair_*` targets. Here provenance is **resolved at plan
time** — every target is `native:`, on one restart file — so the failure mode is not mitigated,
it is **absent** (§7).

---

## 1. What r1 left open, and why nothing here needs new code

| item | blocker (results §11.2) | this wave's answer |
|---|---|---|
| measured pin BU on the optimum | `enable_pin_burnup` is not a deck knob; `WaveVerifier._default_factory` hard-codes `False` (`lpopt/search/verify.py:851`), so **all 95** labelled r1 rows have `max_pin_burnup = null` and `is_deliverable` reports `unknown_axes = ("max_pin_burnup",)` | `pinbu_wave.py` builds the verifier with `curriculum.make_pin_burnup_verifier` (`lpopt/curriculum.py:1017`), which sets it at `:1097` — the only reachable path to the `%EDT_OPT ipin=1` `MAS_PPI` edit |
| `F_xy` re-verification | `_maybe_post_verify` (`campaign.py:2874`) never ran; `post_verify_top_k = 5` produced nothing | 20 exact replays of stored r1 records, with `keep_success=true` retaining the final-cycle `MAS_OUT` that carries `FXYP` |
| `F_xy` of the `F_r`-record family | those rows predate the `F_xy` switch; `f_xy` is null and no campaign will ever backfill it | 5 exact replays of `F_r`-era cores of the **same cell**, retained `MAS_OUT`, scanned offline → **first** `f_xy` labels for that family |

**Why a replicate can run here at all.** `produce` dedups on
`record_id = sha256(canonical_pattern | library_id | case_pair | deck_knobs)`, replayed from the
ledger **and** the store at start (`produce.py:_reconstruct`), so a produce stratum cannot re-run
a stored pattern. `pinbu_wave.py run` never enters that path: it builds `WaveEntry` objects
straight from the manifest and calls `WaveVerifier.evaluate_wave` (`pinbu_wave.py:481-512`).
There is nothing to bypass. The identity check runs the **opposite** way
(`pinbu_wave.py:495-503`): before any MASTER call it mints
`compute_record_id(pattern.canonical(), library_id, case_pair, PRODUCE_DECK_KNOBS)` and **skips
the chain unless it reproduces the manifest's `record_id`**. `PRODUCE_DECK_KNOBS` is the module
**constant** `"ga80_produce"` (`lpopt/search/verify.py:74`), not a hash of this deck, so
`keep_success`, `harvest_maps` and `workers` cannot move the id.

**Pre-verified offline, on the coordinator, before this document was hashed:** all 25 targets
mint their own stored `record_id` (**0 drift**, checked twice — once in the manifest builder and
again by the harness's own gate), and all 25 resolve their paramA assets at
`fallback_level = 0`; `pinbu_wave.py run … --dry-run` printed **25 entries, 0 `[SKIP]`**. §9.

---

## 2. The 25 targets

All 25 are **paramA**, **`T6_T4` / feed 121**, `converged = True`, `valid = True`, joint-clean
(`cbc_max ≤ 1600`, `f_q ≤ 2.41`, `|AO| ≤ 0.30`), and all 25 carry the **same** stored restart,
`native:MAS_RST.APRQ_10_0615.11`. Selection was made on the coordinator, read-only, and is
frozen in the manifest. Predictions were pinned with **`data/models/s1j`** — the r1 campaign's
own serving champion (11th champion, arm-3 direct `f_xy` head) — through `pinbu_wave._score`,
the same serve path the plan step uses, so the manifest stays byte-compatible with
`pinbu_analyze.py`. `s1j`'s G4 gate **FAILED**, so `fxy_head.serve_sigma` is **barred**: the
manifest pins `predicted_f_xy` and leaves `predicted_f_xy_sigma` null. That is the same serving
contract r1 itself ran under.

### 2.1 `minfxy_r1_top20` (role `delivery`) — the pin-BU measurement

**Rule:** the 20 lowest **measured** `F_xy` rows of campaign `fpcamp_minfxy_t6t4_f121_r1` that
are converged, valid and joint-clean; ties broken on `record_id`. 94 of the campaign's 95
labelled rows are joint-clean, so the gate costs almost nothing and no gate was relaxed.
**Ranks 1–5 are exactly the top-5 named in results §11.3.** Span `F_xy` **1.5322 – 1.5890**.

| # | record | stratum | `F_xy` | `F_r` | `F_q` | CBC | \|AO\| | cyclen | maxassy | pred pin | pred `F_xy` |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `bf3a70b20e50` | exploit | **1.5322** | 1.4857 | 1.853 | 1337.4 | 0.024 | 622.10 | 55.2 | 66.59 | 1.5724 |
| 2 | `d06a56d6b928` | explore | 1.5337 | 1.5098 | 1.880 | 1354.8 | 0.024 | 622.97 | 55.5 | 67.77 | 1.6370 |
| 3 | `8b5221442ce8` | exploit | 1.5490 | 1.4912 | 1.860 | 1325.3 | 0.024 | 621.62 | 55.0 | 66.50 | 1.5607 |
| 4 | `e1495cbe169e` | exploit | 1.5577 | 1.5076 | 1.877 | 1337.3 | 0.024 | 622.29 | 55.2 | 66.58 | 1.5650 |
| 5 | `2613a46a257a` | exploit | 1.5631 | 1.4966 | 1.873 | 1341.5 | 0.023 | 623.09 | 55.3 | 66.68 | 1.5610 |
| 6 | `cbed1e74dee6` | exploit | 1.5689 | 1.5210 | 1.902 | 1324.9 | 0.024 | 621.46 | 55.0 | 66.49 | 1.5503 |
| 7 | `89a971f00758` | exploit | 1.5731 | 1.5238 | 1.904 | 1324.3 | 0.024 | 621.36 | 55.0 | 66.63 | 1.5591 |
| 8 | `70dbb5fea765` | exploit | 1.5777 | 1.4769 | 1.832 | 1344.6 | 0.024 | 624.02 | 55.4 | 66.65 | 1.5607 |
| 9 | `3a76842799f0` | exploit | 1.5783 | 1.4867 | 1.815 | 1344.9 | 0.025 | 624.12 | 55.4 | 66.51 | 1.5638 |
| 10 | `50f593b80bdf` | exploit | 1.5783 | 1.4780 | 1.832 | 1348.3 | 0.025 | 624.07 | 55.5 | 66.52 | 1.5642 |
| 11 | `3615e5f5bcd6` | exploit | 1.5817 | 1.4791 | 1.837 | 1345.5 | 0.025 | 624.03 | 55.4 | 66.52 | 1.5606 |
| 12 | `016c7fd4e827` | exploit | 1.5819 | 1.4788 | 1.825 | 1342.8 | 0.025 | 624.07 | 55.4 | 66.53 | 1.5604 |
| 13 | `0dd07dd42c1d` | exploit | 1.5832 | 1.4826 | 1.840 | 1346.3 | 0.025 | 624.06 | 55.4 | 66.73 | 1.5580 |
| 14 | `98acafa3e63d` | exploit | 1.5839 | 1.4830 | 1.841 | 1345.7 | 0.024 | 624.03 | 55.4 | 66.68 | 1.5580 |
| 15 | `a4123b988e5a` | exploit | 1.5842 | 1.4826 | 1.841 | 1344.7 | 0.024 | 623.86 | 55.4 | 66.82 | 1.5584 |
| 16 | `094c7629068e` | exploit | 1.5871 | 1.4911 | 1.818 | 1344.2 | 0.025 | 624.14 | 55.4 | 66.49 | 1.5645 |
| 17 | `b456608c8ef2` | exploit | 1.5876 | 1.4927 | 1.865 | 1331.3 | 0.025 | 622.70 | 55.1 | 66.41 | 1.5700 |
| 18 | `de57276dc848` | exploit | 1.5883 | 1.4879 | 1.850 | 1344.0 | 0.024 | 624.03 | 55.4 | 66.67 | 1.5695 |
| 19 | `e591bddf9139` | exploit | 1.5883 | 1.4919 | 1.829 | 1343.4 | 0.025 | 624.06 | 55.4 | 66.51 | 1.5609 |
| 20 | `e88e7be53e5c` | exploit | 1.5890 | 1.5101 | 1.887 | 1338.2 | 0.023 | 622.79 | 55.2 | 66.73 | 1.5620 |

### 2.2 `frera_fxy_measure` (role `calibration`) — `F_xy` on the `F_r`-record family

**Rule:** the `F_r` record core `4d70ab6f` (1.4605, `batchswap_enum_T6T4`) and the λ-opt
`188c9a33` (1.4749, `fpcamp_minfr_T6T4_r8`) — the two cores results §11.3 names — **plus** the
3 next lowest-`F_r` joint-clean rows of the **same `T6_T4`/f121/paramA cell whose `f_xy` is
null**, excluding every `F_xy`-era campaign; ties on `record_id`. Span `F_r` **1.4605 – 1.4749**,
i.e. entirely **below** the `min_fxy` top-20's `F_r` span (1.4769 – 1.5238) — which is the point:
these are the cores the `F_r` objective preferred.

| record | campaign | `F_r` | `F_q` | CBC | \|AO\| | cyclen | maxassy | `f_xy` | pred pin | pred `F_xy` |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `4d70ab6f75d4` | batchswap_enum_T6T4 | **1.4605** | 1.807 | 1302.7 | 0.024 | 618.02 | 55.2 | **null** | 66.54 | 1.5824 |
| `0a83b6547650` | batchswap_enum_T6T4 | 1.4612 | 1.829 | 1299.0 | 0.023 | 618.79 | 55.5 | **null** | 66.58 | 1.6044 |
| `456199b370ad` | batchswap_enum_T6T4 | 1.4616 | 1.810 | 1303.9 | 0.024 | 618.09 | 55.0 | **null** | 66.57 | 1.5962 |
| `eaf914e573b2` | batchswap_enum_T6T4 | 1.4648 | 1.828 | 1311.9 | 0.024 | 618.46 | 55.0 | **null** | 66.53 | 1.5807 |
| `188c9a338d9f` | fpcamp_minfr_T6T4_r8 | **1.4749** | 1.823 | 1355.8 | 0.024 | 625.46 | 55.3 | **null** | 66.76 | 1.5997 |

**These five are NOT replicates on `F_xy`.** Their `f_xy` is null by construction, so they
contribute a **first label**, not a `|ΔF_xy|`. They *are* full replicates on the scalar axes
(`f_r`, `f_q`, `cbc_max`, `cyclen`, `ao_abs`, `max_assembly_burnup`) and they carry a full
predicted-pin value, so they stay inside `pinbu_analyze.py`'s existing `calibration` taxonomy
(`pinbu_analyze.py:190, 221`) with no analysis-code change. Their predicted-pin **span is narrow
(66.53–66.76)**, so — stated honestly up front — this set is a *replicate* set, not a
calibration-curve set; the curve fit `pinbu_analyze.py` will emit over it is degenerate and must
be ignored in the readout.

### 2.3 Both sets are exact replays

All 25 chains replay a stored pattern under the identity gate of §1, so **all 25** contribute to
the scalar determinism readout and **all 25** produce a pin value. The A/B split is what each set
was *selected* for, not what it can report.

---

## 3. Deck

`pinbu_wave_minfxy_r1_199.inp` is a clone of `pinbu_wave_fxyera_r1_199.inp` with **four deltas
and no others** (non-comment `diff` in §9). **All four are inert on the run path**:

1. `[model] model_dir = "data/models/s1j"` (was `s1i`) — no checkpoint is loaded here; set to the
   campaign's own serving champion so the deck matches the manifest's `model_dir`.
2. `[model] library_id = "paramA"` (was `"ga80"`) — a resolver **default** only; `pinbu_wave.py`
   passes the per-core `library_id` explicitly (`_verifier_for`). Set to the value the wave
   actually uses instead of one it never reaches.
3. `[case] pair = "T6_T4"`, `feed = 121` (was `N1_N2` / 113) — `pinbu_wave.py run` **never reads
   `[case]`**; every `CaseKey` comes from the manifest's own rows. Set to the wave's real cell so
   the deck cannot mislead a reader.
4. header / `[flow] title` text.

`[master]`, `[produce]`, `[verify]` and `[design]` — the only four sections this harness acts on
— are **byte-identical** to the precedent deck apart from comments. `[model] cond_schema` stays
`"v8"`: `s1j/PROMOTION.md` states the deck key does not move from `s1i`.

`[master] keep_success = true` is **load-bearing**: `F_xy` is MASTER's `FXYP`, printed **only in
`MAS_OUT`**, and it is not on `WaveFom` — so `pinbu_wave.py` cannot report it and the replay's
`F_xy` must be read **offline** from the retained final-cycle `MAS_OUT`, with the same scanner
that produced `fxy_backfill_199_pinbu_wave_fxyera_r1_20260830.csv`. `keep_success = false` purges
100% of those dirs (`pinbu_rodavg_20260820.md`). **No retained `MAS_OUT`, no `F_xy` at all** —
and for group B that loss is permanent within this wave, because its `F_xy` is a first label and
not recoverable from any stored column.

`harvest_maps = true` does two things, both wanted: `curriculum.py:1075` widens `keep_success` to
`keep_success OR harvest` (retention no longer rests on one knob), and `_factory` installs
`HarvestingEquilibriumEvaluator` so `WaveOutcome.maps` carries the final EDIT5 map
(`curriculum.py:1100-1104`). **Honest limit:** `pinbu_wave.py._outcome_record` does not persist
`outcome.maps`, so this wave writes **no** `maps.npz`. That costs nothing — all 25 targets
already carry a non-null `maps_key` — and the artifact that matters is the retained
`MAS_SUM`/`MAS_OUT` on disk.

`[verify] package_root = "FEASIBLE_PACKAGE"` is retained but **never resolved** (no ga80 target);
the launcher reports its absence rather than refusing.

**Run dir stays relative** (`--run-dir runs/pinbu_wave_minfxy_r1`). Nothing in the deck names a
drive, so if `runs/` is a junction to another volume the wave follows it with no edit; the
launcher and the status probe resolve `runs/`'s real target and report free space on **that**
volume, not on `C:`.

---

## 4. Registered marks (scored strictly; a failed control is a refusal, never a downgrade)

### Primary

- **M1 — pin axial peak on the r1 `F_xy` frontier → the DELIVERY verdict for the r1 optimum.**
  Reported: the **rate of measured `max_pin_burnup` ≤ 80.0 GWd/tU over the 20
  `minfxy_r1_top20` cores**, the ≤ 78.0 acquisition-gate rate, the full 20-core distribution,
  **and, called out on its own line, the measured pin of rank 1 `bf3a70b2` — the number that
  turns the round's optimum from `unknown_axes = ("max_pin_burnup",)` into a deliverable or
  not.**
  **Pre-registered prediction: 20/20 = 100% measure ≤ 80, and 20/20 measure ≤ 78.**
  Basis, and why this bar is far stronger than the `F_xy`-era wave's ≥ 80%: s1j predicts
  **66.41 – 67.77** on all 20 (median 66.58), i.e. **≥ 12.2 GWd/tU of headroom** to the 80
  limit, against a measured feed-121 head bias of **+3.23 with MAE = bias** — no negative
  residual anywhere in that slice, n = 13, CI [+2.39, +4.07]
  (`pinbu_wave_fxyera_r1_results_20260830.md` §5). The head over-predicts here; measured values
  are expected near **63–65**. Stored `max_assembly_burnup` is 55.0–55.5 on all 20, a low-burnup
  cell. Even a **−5.0** residual (worse than any feed-121 residual on record, and worse than the
  worst residual of any feed in that wave, −5.01) leaves every core at ≤ 72.8.
  **Falsifier:** any single core measuring > 80 falsifies M1. It would mean the pin head's
  feed-121 conservatism does not extend to `F_xy`-selected cores at this cell, and that
  `minfxy_pin_bu_limit = 78` must be re-derived before any r2 deck is written.

- **M2 — `F_xy` determinism.** For every `minfxy_r1_top20` replay that reproduces its stored
  `f_r`/`cyclen`/`cbc_max` inside `TOL_F_R`/`TOL_CYCLEN`/`TOL_CBC` (0.002 / 0.5 / 2.0) **and**
  keeps its restart provenance, report `|F_xy_replay − F_xy_stored|` read from the retained
  `MAS_OUT`. **Pre-registered prediction: `|ΔF_xy| ≤ 0.002` on every such chain (max over 20
  ≤ 0.002).** Basis: `pinbu_wave_fxyera_r1` measured `|ΔF_xy| = 0.0000` on **40/40**, including
  on 8 chains seeded from a *different* restart. **Falsifier:** any chain exceeding 0.002 means
  r1's `f_xy` labels carry replay noise of that order, and the 1.5322 → 1.65 margin — and every
  `F_xy` number in the r1 results — must be widened by the measured spread before use. This mark
  is also the substitute deliverable for the `post_verify_top_k = 5` re-verification that r1
  never ran (results §11.2): it re-verifies the top 20, not the top 5.
  **Group B is excluded from M2 by construction** (`stored.f_xy` is null); it is not a
  "missing" replicate and must not be reported as one.

- **M3 — chain integrity / provenance.** **Pre-registered prediction: 25/25 converged and
  25/25 `provenance_ok`.** Unlike the precedent (which registered the same and measured 32/40),
  this is **structural, not hopeful**: provenance was resolved at plan time and every target is
  `native:` on the single restart `MAS_RST.APRQ_10_0615.11`, so no promoted-cache state can move
  a resolution — see §7. The coordinator dry-run already resolves all 25 at
  `fallback_level = 0`. **Falsifier:** any `provenance_ok = False` is a genuinely new failure
  mode (not the known `pair_*` drift) and must be reported as such, not absorbed.

- **M4 — `F_xy` of the `F_r`-record family vs the r1 optimum.** For each of the 5
  `frera_fxy_measure` cores, report the **first measured `F_xy`**, read offline from the
  retained `MAS_OUT`, against the r1 optimum's **1.5322**. The headline question, in the words
  of results §11.3: *does the `F_r` record `4d70ab6f` beat 1.5322 on `F_xy`?*
  **Pre-registered prediction: NO — all five measure `F_xy` > 1.5322; point estimate for
  `4d70ab6f` ≈ 1.59.**
  Basis, two independent estimators:
  * **s1j's own head** predicts `4d70ab6f` at **1.5824**, and all five between 1.5807 and
    1.6044 — each above **19 of the 20** `minfxy_r1_top20` predictions. Bias-corrected on this
    wave's own 20 labelled rows (residual pred−meas: bias −0.0075, **median −0.0181**, MAE
    0.0238, sd 0.0309), the five land at **1.588 – 1.612** (mean-corrected) or **1.599 – 1.623**
    (median-corrected).
  * **A model-free ratio estimator.** The `F_xy`/`F_r` ratio over the 20 labelled cores is
    median **1.0640**, span 1.0158 – 1.0697. Applied to `4d70ab6f`'s `F_r` of 1.4605 the median
    ratio gives **1.554**.
  **Honest caveat, registered so the mark stays falsifiable:** the ratio estimator's *lower*
  tail (1.0158 × 1.4605 = **1.484**) does **not** exclude a win, and s1j's `f_xy` head was
  measured **ranking-incompetent within this cell** (results §6.3: joint-clean ρ +0.016,
  exploit-slot ρ −0.383). So the prediction rests on a level argument, not a ranking one, and a
  falsification here is entirely plausible.
  **Falsifier and its consequence:** if any of the five measures `F_xy` ≤ 1.5322 — in particular
  if `4d70ab6f` does — then the `F_r`-era search already held the cell's `F_xy` record, r1's
  headline gain (`1.5491 → 1.5322`) is **not** a gain over the cell's true best, and the
  marginal value of the `min_fxy` objective at this cell is zero or negative. That is a
  decision-grade result and it directly gates the r2 cell choice registered in results §11.4-5.

### Secondary (measured and reported; failure does not invalidate the wave)

- **M5 — pin-head skill, and a direct replication of the feed-121 bias.**
  `pin_error_pred_minus_meas` pooled and by group. This wave puts **25 chains at feed 121**
  against the 13 that produced the precedent's `+3.230` (CI [+2.39, +4.07], MAE = bias). It is
  therefore a clean, larger-n replication of the single most actionable number that wave
  produced — the one behind its recommendation to replace `minfr_pin_bu_limit = 78` with a
  feed-stratified margin. Reported: bias, MAE, sd, 95% CI, and whether the CI overlaps
  [+2.39, +4.07].
- **M6 — scalar determinism.** Max `|Δ|` over the 25 chains on `f_r`, `f_q`, `cbc_max`,
  `cyclen`, `ao_abs`, `max_assembly_burnup`, against the precedent's 0.000000 / 0.000100 /
  0.020000 / 0.006000 / 0.000100 / 0.003000.
- **M7 — `F_xy`/`F_r` ratio at low `F_r`.** Group B extends the measured (`F_xy`, `F_r`) pair
  set **below** `F_r` 1.4769 for the first time at this cell. Pinned now from the manifest: over
  the 20 labelled targets the stored ratio runs **1.0158 – 1.0697, median 1.0640**. Group B's
  five measured ratios are reported against that span. **Caveat, as in the precedent:** this
  cell produced no converged core with `F_r` in [1.50, 1.65] below 1.5238, so the `S1`
  wall-location interval [1.04, 1.08] is again not scorable in the region it was registered for.
- **M8 — `f_xy` head level skill on unseen cores.** The 20 group-A residuals are pinned in §2.1
  (bias −0.0075, MAE 0.0238); the readout restates them against the promotion bar
  `G2′ MAE < 0.0767` (`s1j/PROMOTION.md`), on this cell only. Reported, not a gate.

---

## 5. Budget

| item | value |
|---|---|
| chains | 25 (20 + 5), one per core, no replication beyond the registered sets |
| workers | 12 (`[produce] workers`), all-P box, `use_all_cores = true`, `host_reserve = 0` |
| per-chain wall, precedent | median 349.6 s, max 638.5 s (40 chains, 12 workers, `pinbu_wave_fxyera_r1`, 2026-08-30) |
| batches | ⌈25/12⌉ = **3** (the precedent ran 4 for 40 chains in 2,104 s) |
| **expected wall** | **~0.35 h** (25/40 × 2,104 s ≈ 1,315 s, plus staging) |
| **registered cap** | **1.0 h.** Exceeding it is reported, not silently absorbed |
| hard bound | `chain_timeout = 3600 s` × 3 batches |
| disk | 25 retained final-cycle dirs; launcher refuses below 10 GB free **on the volume `runs/` resolves to** |
| RAM | launcher refuses below 30 GB free (12 concurrent chains) |
| store writes | **none during the run** (`fuel_types.parquet` is read-only); merge is a separate `patch` step, dry-run first, backed up |

This cell's cycles are short (cyclen 618–625 EFPD, `n_cycles` 10–12), the same regime the
precedent's paramA subset ran in, so the linear scaling above is not an extrapolation.

---

## 6. Ordering: this wave must not start while `intervention_wave_r1` is running

`intervention_wave_r1` (Campaign A round 1, 800 paid chains, 24 workers, ~14.4 h registered)
owns box 199's MASTER queue, and a ga80 resume leg
(`resume_intervention_wave_r1_ga80_199.bat`) may still be pending. Two MASTER queues on one box
is how a wave silently loses its cadence — and §5's budget claim is calibrated on an **empty**
box. The launcher therefore **refuses** on any `master4.0m4_r1` process **or** any `python.exe`
whose command line matches `lpopt|ablation|batchswap|pinbu|produce|intervention|mesh`, and prints
the offending command lines so the operator can see what is holding the box.

**`intervention` and `mesh` are new in this regex.** The `pinbu_wave_fxyera_r1` launcher gated on
`lpopt|ablation|batchswap|pinbu|produce`, which would **not** have matched
`python.exe -u intervention_wave.py run …` — that wave imports `ablation_wave` but does not run
it. That is a latent hole in the precedent, closed here.

---

## 7. The precedent's M3 failure, and why it cannot happen here

`pinbu_wave_fxyera_r1_results_20260830.md` §1: 40/40 converged, 40/40 determinism-ok, but
**32/40 `provenance_ok`**. The drift ran **opposite** to the registered risk. Its §7 registered
a *pruned* promoted cache making a `promoted:` target fall back to `pair_*`; what happened is
that the cache had **grown** — r1 itself populated those cells after the store rows were written
— so **all 8** targets whose store row recorded a `pair_ecore:`/`pair_feed:` restart resolved a
`promoted:` restart instead and were refused by `pinbu_wave.py patch`. All 18 `promoted:` and
all 14 `native:` targets resolved correctly. The launcher's presence check could not see it:
the cache was there, it was bigger.

**Two corrections, applied here:**

1. **Provenance resolved at PLAN time — the real fix.** A row is eligible for this wave only if
   its stored `restart_provenance` begins `native:`. All 25 do, on **one** restart file,
   `MAS_RST.APRQ_10_0615.11`, which resolves out of `data\design\package` and **never** out of
   `data\produce\promoted`. There is therefore no promoted-cache state — grown, pruned or
   unchanged — that can move a resolution, and the precedent is direct evidence: under exactly
   the cache growth that broke its 8 `pair_*` targets, **14/14 of its `native:` targets resolved
   correctly**. The manifest records this as a first-class block (`provenance_plan`:
   `n_promoted_dependent = 0`, `n_pair_fallback_dependent = 0`, `expected_provenance_ok = 25`),
   and the launcher re-asserts it by **reading the manifest on disk** and refusing if any target
   is not `native:` — so a later edited manifest cannot quietly reintroduce the dependency.
   The coordinator dry-run resolves all 25 at `fallback_level = 0`, which the precedent's
   `promoted:` targets could not do here (they fell back to `pair_ecore:`, §7 of that document).
   **This wave's resolution is box-independent; the precedent's was not.**
2. **The cell-count gate that document recommended** (§8, first bullet: *"a launcher gate on
   promoted-cache **cell count**, not mere presence, which would have caught §1's drift before
   0.58 h was spent"*). The launcher counts the directories under `data\produce\promoted` and
   compares against a pinned `$wantPromCells`. Because this wave has **zero** promoted-dependent
   targets the count is a **change detector on the resolution environment**, not a dependency —
   but it is still gated, because a cache that moved between stamping and arming means the box's
   produce state moved, and that is worth refusing 25 chains over.
   **It ships FAIL-CLOSED.** `$wantPromCells = -1` is the unstamped sentinel and the launcher
   **refuses**, printing the observed count. The coordinator cannot observe 199's cache from
   here, so the value is deliberately **not guessed**: it is stamped in §9 step 0 below.

**Registered mix, for scoring:** `native` 25, `promoted` 0, `pair_ecore` 0, `pair_feed` 0;
**1** distinct restart file. Expected `provenance_ok` = 25/25.

---

## 8. Hashes (pinned before launch; the launcher gates on the first four and the store)

| artifact | bytes | sha256 |
|---|---:|---|
| `pinbu_wave_minfxy_r1_199.inp` | 10,213 | `1C27E3971DE6010E95477A1451FD380E8DEE2BAD992D177F0517DF19030E6C88` |
| `data/reports/pinbu_wave_minfxy_r1_manifest.json` | 60,019 | `BC035D442891A79A640E0BFD7A3FAAB58B24A1D5E7D5A7F65494B268A3C35041` |
| `pinbu_wave.py` | 36,222 | `5B3688CFAD684E9E837910F8842F68A2F2C21F931052DD52DE98262BA3581047` |
| `run_pinbu_wave_minfxy_r1_199.bat` | 2,117 | `A23502217AFB397A3C65B9822BFDA3CAADA8CBA14999D301FA89C0A45AFDDB3F` |
| `data/store/records.parquet` | 22,570,584 | `255F0E41707CB4EF64D843FD19DB81531C12AB3A969F6F8F06C87E0AF5561A51` |
| `data/store/fuel_types.parquet` | 64,343 | `FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137` |
| `launch_pinbu_wave_minfxy_r1_199.ps1` **(UNSTAMPED)** | 12,321 | `4D7FE951F3A91503AB71A79F22C4546848DE419ADB853AC74DD5BE2669C0A05A` |
| `launch_pinbu_wave_minfxy_r1_199.ps1` **(STAMPED)** | — | *to be written here at §9 step 0* |
| `status_pinbu_wave_minfxy_r1_199.ps1` | 6,314 | `F347CBDC04441C278D26B36A607234A8B040100659C42C51813ADE6558F02C3C` |

**The launcher hash above is the UNSTAMPED artifact** (`$wantPromCells = -1`, refuses). §9 step 0
changes exactly one integer in it; the resulting hash **must be written back into this table**
before arming, and the run report must quote the stamped value. Nothing else in the launcher may
change at that step.

`pinbu_wave.py` is byte-identical to the harness the precedent gated on, so the two waves share
one harness provenance. The store gate is a **sha256** gate: the manifest pins 25 stored
`f_xy`/`f_r`/`cyclen`/`cbc_max` labels and those labels *are* the determinism reference, so a
store that differs at all leaves the wave unscored. The value is the coordinator's **current
local** store, 75,893 rows = the 75,793 of the `F_xy`-era merge plus the 100 rows of
`fpcamp_minfxy_t6t4_f121_r1`. **Note for the merge step (results §11.3, D5):** `pinbu_wave.py
patch` will change the store sha again, so the r1 results §9.1 row and any launcher `$wantSha`
must be re-stamped after this wave's patch.

---

## 9. Pre-launch validation already performed (coordinator, read-only, no MASTER)

| check | command | result |
|---|---|---|
| deck is a legal `LpoptConfig` and its assets | `python -m lpopt check --input pinbu_wave_minfxy_r1_199.inp` | `0 PASS, 2 FAIL, 1 SKIP` — **both FAILs are coordinator-box absences**: `master.executable` (`C:\DeCART_MASTER\BIN\master4.0m4_r1.exe`, lives on 199) and `verify.package_root` (`FEASIBLE_PACKAGE`, the ga80 kit dir, absent here and unused by this wave). `pinbu_wave_fxyera_r1_199.inp` returns the **identical** two FAILs, so the result is host state, not deck state. |
| deck delta vs precedent | `diff` on non-comment lines, `pinbu_wave_fxyera_r1_199.inp` → this deck | exactly the four deltas of §3 (`model_dir`, `library_id`, `[case] pair`/`feed`, title) — `[master]`, `[produce]`, `[verify]`, `[design]` unchanged |
| manifest schema | `schema == "pinbu_wave_prereg_v1"`, 25 targets, 25 unique `record_id`, 2 groups, all `library_id == "paramA"`, all `case_pair == "T6_T4"`, all `feed == 121` | pass |
| **record_id identity, all 25** | `compute_record_id(pattern.canonical(), library_id, case_pair, PRODUCE_DECK_KNOBS)` vs stored, checked in the manifest builder **and** again by the harness gate | **0 drift** (`record_id_minted_ok = true` on 25/25) |
| **asset resolution, all 25** | `python pinbu_wave.py run --plan <manifest> --deck <deck> --run-dir <tmp> --dry-run` | **25 entries, 0 `[SKIP]`**, one paramA verifier, `workers = 12`, `fallback_level = 0` on 25/25, restart provenance `['native:MAS_RST.APRQ_10_0615.11']` |
| top-5 identity | manifest ranks 1–5 vs results §11.3 table | exact match: `bf3a70b2` 1.5322, `d06a56d6` 1.5337, `8b522144` 1.5490, `e1495cbe` 1.5577, `2613a46a` 1.5631 |
| launcher / probe syntax | PowerShell AST `Parser::ParseFile` | both `PARSE_OK` |
| launcher manifest gate | `ConvertFrom-Json` on the manifest under Windows PowerShell 5.1 (it contains bare `NaN` in `predicted_pin_band`, as the precedent's does) | parses; `n_targets = 25`, non-native targets = **0**, restarts = `native:MAS_RST.APRQ_10_0615.11` |

---

## 10. Execution order (to be run only on the coordinator's instruction)

```
# 0. STAMP the promoted-cache cell-count gate (read-only; the launcher REFUSES until done)
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_pinbu_wave_minfxy_r1_199.ps1"
#    -> read PROMOTED_CELLS n ; set $wantPromCells = n in launch_pinbu_wave_minfxy_r1_199.ps1
#    -> re-hash the launcher and write the new sha into section 8 of this document
#    -> this step also confirms PROCS master=0 python_intervention=0 (section 6)

# 1. ARM (the ONLY command that starts MASTER)
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_pinbu_wave_minfxy_r1_199.ps1"

# 2. WATCH (read-only, starts nothing)
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_pinbu_wave_minfxy_r1_199.ps1"

# 3. MERGE the measured pin column -- dry run FIRST, always
python pinbu_wave.py patch --results runs/pinbu_wave_minfxy_r1/pinbu_wave_results.jsonl --dry-run
python pinbu_wave.py patch --results runs/pinbu_wave_minfxy_r1/pinbu_wave_results.jsonl --tag pinbu_minfxy_r1_20260830
```

A partial run **resumes**: re-invoke the `.bat`, never the launcher (the launcher deletes the run
dir, and the results JSONL in it is the only resume ledger).

**M2 and M4 are not produced by any of the four commands above.** Both are an offline pass over
`runs/pinbu_wave_minfxy_r1/paramA/master_work/**/MAS_OUT*` with the r1 `FXYP` scanner, joined to
the manifest on `record_id` — compared to `stored.f_xy` for `minfxy_r1_top20` (M2), and reported
as first labels for `frera_fxy_measure` (M4). That readout, and the results report
`data/reports/pinbu_wave_minfxy_r1_results_<date>.md`, are the next deliverable — not part of
this one.

**Deliverables of this pre-registration, all present and hashed, nothing launched:** deck,
manifest, launcher trio (`launch_` / `run_` / `status_pinbu_wave_minfxy_r1_199`), this document.

**STAMP 2026-08-30 (step 0):** HOST_199 `data\produce\promoted` cell count = 8 → `$wantPromCells = 8`; launcher re-hashed sha256 = `A28A73FB65CE8C07CC876E5426DEE671D2968BA07BA9556FAACEC22B8FED3338`.

**STAMP 2026-08-30 22:xx:** store pin → `73701E33F07291E17609BA30D025E2A5B7A423FEB69F08D23DE4EC23EBE0C85F` (76,693행, intervention r1 병합 반영); launcher 재해시 `6766169810F6E06CD151EF0CABE58211ED09D0CFAF64B1FFF8AF8997B523DCDD`.

**STAMP 2026-08-30 22:xx (2):** store byte length pin 22,570,584 → 22780281; launcher 재해시 `ACC6F98364D28377382588EEC7DD5CC066ABA35567CB0EE7975B120D2458E8E0`.
