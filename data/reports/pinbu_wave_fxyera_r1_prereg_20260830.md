# Pre-registration — phase-2 pin-burnup + F_xy determinism wave (F_xy-era r1, box HOST_199)

**Date** 2026-08-30 · **Deck** `pinbu_wave_fxyera_r1_199.inp` · **Run dir** `runs/pinbu_wave_fxyera_r1`
**Harness** `pinbu_wave.py run` (NOT `lpopt optimize`, NOT `lpopt produce`)
**Manifest** `data/reports/pinbu_wave_fxyera_r1_manifest.json` (`schema: pinbu_wave_prereg_v1`, 40 targets)
**Registered by** `data/reports/produce_fxyera_r1_prereg_20260829.md` §6 ("Phase-2 plan … 40 chains")
**Store at write time** `data/store/records.parquet`, 75,793 rows (r1 merged), sha256 pinned in §8
**Status** WRITTEN BEFORE ANY MASTER CALL. Hashes pinned in §8. **Nothing launched.**

---

## 0. One-paragraph statement

r1 has drained and merged. It delivered the programme's first `f_xy` labels — 540 of them,
scanned out of retained `MAS_OUT`. Two things §6 pre-registered as *phase 2* are now due, and
`pinbu_wave.py` does both in one pass: (a) measured `max_pin_burnup` on the cores r1 actually
put nearest the `F_xy` wall, and (b) exact re-runs of stored r1 records, which — because
`keep_success=true` retains the final cycle's `MAS_OUT` — are simultaneously the first
**determinism measurement for `F_xy` itself**. 40 chains, one deck, no code change, no new
strata. This document also settles the question that gated the readout of r1: **does the 3.0
`above_ceiling` cut truncate real high-`F_xy` cores?** It does not (§2), so the 540-label set
is the whole of r1's physically meaningful `F_xy` support and the selection below is not
biased by the cut.

---

## 1. What §6 registered, and why nothing here needs new code

§6 stated the two blockers precisely, and both survive verification against the current tree:

| item | blocker | this wave's answer |
|---|---|---|
| pin burnup | `enable_pin_burnup` is not a deck knob; `WaveVerifier._default_factory` hard-codes `False` (`lpopt/search/verify.py:851`) | `pinbu_wave.py` builds the verifier with `curriculum.make_pin_burnup_verifier` (`lpopt/curriculum.py:1017`), which sets it at `:1097` |
| replicate QC | `produce` dedups on `record_id`, replayed from ledger **and** store at start (`produce.py:_reconstruct`), so a produce stratum cannot re-run a stored pattern | `pinbu_wave.py run` never enters that path: it builds `WaveEntry` objects straight from the manifest and calls `WaveVerifier.evaluate_wave` (`pinbu_wave.py:481-512`) |

**How the `keep` deck "bypasses" the dedup — verified, and it is worth stating exactly, because
the mechanism is the opposite of what the phrase suggests.** There is nothing to bypass: no
deck knob disables dedup and `pinbu_wave_keep_199.inp` does not try to. The dedup lives in
`produce.py`, on a code path this harness does not call. What the harness *does* have is an
**identity gate that runs the other way** (`pinbu_wave.py:495-503`): before any MASTER call it
mints `compute_record_id(pattern.canonical(), library_id, case_pair, PRODUCE_DECK_KNOBS)` and
**skips the chain unless it reproduces the manifest's `record_id`**. So the wave is *required*
to be an exact replica of a stored row, not merely permitted to be one. The knob that could
have broken this — `PRODUCE_DECK_KNOBS` — is the module **constant** `"ga80_produce"`
(`lpopt/search/verify.py:74`), not a hash of the deck, so `keep_success`, `harvest_maps` and
`workers` cannot move the id. The single delta of `pinbu_wave_keep_199.inp` over
`pinbu_wave_199.inp` is `[master] keep_success = true`; that is a *retention* change, not a
dedup change.

**Pre-verified offline, on the coordinator, before this document was hashed:** all 40 targets
mint their own stored `record_id` (0 drift), and all 40 resolve their paramA assets —
`python pinbu_wave.py run --plan … --deck … --dry-run` printed 40 entries, 0 `[SKIP]`. §9.

---

## 2. The 3.0 `above_ceiling` cut — ANSWERED: it truncates nothing real

r1's `MAS_OUT` scan produced 1,093 rows in
`data/reports/fxy_backfill_199_produce_fxyera_r1_20260829.csv`; 540 were accepted (`sane=1`)
and merged, 224 finals were **withheld with `reason = above_ceiling`** at a 3.0 garbage
ceiling. The decision question was whether that cut is silently deleting genuine high-`F_xy`
cores. Measured:

| set | n | `F_xy` min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|
| **withheld** (`above_ceiling`) | 224 | **3.0040** | 3.4960 | 3.8682 | 4.4639 | **6.3957** |
| accepted (`sane=1`, merged) | 540 | 1.6926 | 2.1155 | 2.2803 | 2.4740 | **2.9928** |

Also withheld, and unrelated to the ceiling: 312 `nonfinite`, 10 `superseded`, 7
`first_cycle`.

**Verdict — the ceiling is safe, and the 540-label set is complete.** Three reasons, all from
the table:

1. The withheld minimum is **3.0040**, i.e. **1.82×** the `F_xy` licensing limit of 1.65 and
   1.55× the highest *accepted* value. Nothing in the withheld set is within reach of any
   feasibility decision; the withheld median (3.87) is 2.3× the limit.
2. There is no pile-up at the cut. The accepted maximum (2.9928) and the withheld minimum
   (3.0040) sit either side of 3.0 with the distribution continuing smoothly through it —
   the signature of a threshold crossing a genuine tail, not of a clamp manufacturing one.
   Had the parser been clipping, the withheld set would show a spike *at* 3.000.
3. The withheld `F_xya` (axial-integrated companion) runs 2.2677–5.7098 — likewise nowhere
   near feasibility, so the withheld rows are not a case of a bad `F_xy` on an otherwise
   sound core.

**Consequence for r1's reading:** these 224 are unconverged-quality or pathological cores, not
truncated frontier. `S2` of the r1 pre-registration ("r1 produces at least one converged core
with `F_xy` ≤ 1.65") is therefore **falsified on complete data, not on a censored sample**:
the whole-store minimum over 540 labels is **1.6926** (`5cecf2d73b1b`, `fxy_exp_S3T1S5_f125`,
`F_r` 1.6529, CBC 1603.4, cyclen 729.5) — 2.6% above the limit, and 0 cores clear it. That
core is included in this wave (§3, set B) precisely because it is the closest approach on
record.

---

## 3. The 40 targets

All 40 are **paramA** (every r1 stratum is), all are `converged=True, valid=True` with a
non-null `f_xy`. Selection was made on the coordinator, read-only, and is frozen in the
manifest. Predictions were pinned with champion `data/models/s1i` through
`pinbu_wave._score` — the same serve path the plan step uses — so the manifest is
byte-compatible with `pinbu_analyze.py`.

### 3.1 `fxywall_top20` (role `delivery`) — the pin-BU measurement, §6 item 1

**Rule:** the 20 lowest **measured** `F_xy` among r1 rows that are converged, valid and
**joint-clean** (`cbc_max ≤ 1600`, `f_q ≤ 2.41`, `|AO| ≤ 0.30`); ties broken on `record_id`.
42 r1 rows are joint-clean, so "joint-clean where possible" binds fully — no gate was relaxed.
Span `F_xy` **1.7836 – 1.9544**.

| record | stratum | feed | F_xy | F_r | F_q | CBC | \|AO\| | cyclen | pred pin | restart |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7627c10841fe | fxy_exp_S3T1S5_f125 | 125 | 1.7836 | 1.7036 | 2.139 | 1592.7 | 0.029 | 712.7 | 76.8 | promoted |
| 531d5a436d9e | fxy_exp_HGD569_f125 | 125 | 1.8405 | 1.7384 | 2.154 | 1584.6 | 0.026 | 731.6 | 77.7 | promoted |
| 3f77a1f570ea | fxy_bnd_T3T4_f121 | 121 | 1.8454 | 1.7576 | 2.157 | 1167.8 | 0.049 | 595.1 | 70.1 | native |
| e5cd06b2b57f | fxy_exp_HGD569_f109 | 109 | 1.8607 | 1.7602 | 2.201 | 1519.5 | 0.024 | 673.4 | 87.0 | promoted |
| 43d81bf75374 | fxy_exp_HGD569_f125 | 125 | 1.8773 | 1.8042 | 2.243 | 1591.5 | 0.030 | 728.4 | 79.0 | pair_ecore |
| 9ebd5831d8c2 | fxy_ood_T6T4_f109 | 109 | 1.8888 | 1.8089 | 2.224 | 1261.5 | 0.032 | 576.5 | 79.3 | promoted |
| 7c38492e122a | fxy_bnd_T3T4_f121 | 121 | 1.8952 | 1.7770 | 2.292 | 1335.0 | 0.038 | 599.7 | 69.3 | native |
| 4f6e002473b4 | fxy_bnd_T3T4_f121 | 121 | 1.9007 | 1.7981 | 2.248 | 1210.4 | 0.053 | 593.3 | 72.3 | native |
| 6207fd6e8216 | fxy_exp_HGD569_f125 | 125 | 1.9057 | 1.7899 | 2.222 | 1515.7 | 0.027 | 719.3 | 78.1 | promoted |
| 048fedfa1a14 | fxy_exp_T6T4_f121 | 121 | 1.9064 | 1.8095 | 2.278 | 1427.5 | 0.021 | 620.8 | 71.1 | native |
| d0c9a51a1eee | fxy_exp_S3T1S5_f125 | 125 | 1.9090 | 1.8338 | 2.278 | 1571.3 | 0.034 | 722.1 | 77.2 | promoted |
| f0da920dcabd | fxy_exp_HGD569_f109 | 109 | 1.9118 | 1.8112 | 2.262 | 1489.0 | 0.038 | 660.5 | 94.7 | pair_feed |
| bec9339d8845 | fxy_exp_HGD569_f125 | 125 | 1.9168 | 1.7471 | 2.230 | 1520.0 | 0.038 | 719.1 | 79.0 | pair_ecore |
| 8449def1d565 | fxy_exp_S3T1S5_f125 | 125 | 1.9185 | 1.7726 | 2.209 | 1596.0 | 0.031 | 725.9 | 78.3 | promoted |
| b58578254a89 | fxy_exp_HGD569_f125 | 125 | 1.9284 | 1.8182 | 2.257 | 1542.6 | 0.025 | 722.6 | 79.1 | pair_ecore |
| 30f4a07b8343 | fxy_exp_S3T1S5_f125 | 125 | 1.9334 | 1.7932 | 2.229 | 1570.4 | 0.031 | 719.5 | 79.0 | promoted |
| 467cb54ab299 | fxy_exp_T6T4_f121 | 121 | 1.9346 | 1.8376 | 2.345 | 1434.7 | 0.021 | 618.8 | 72.4 | native |
| 6db46706030c | fxy_exp_HGD569_f109 | 109 | 1.9367 | 1.8050 | 2.224 | 1445.0 | 0.032 | 660.7 | 94.2 | pair_feed |
| a0d1e59917d8 | fxy_bnd_S3T1S5_f125 | 125 | 1.9536 | 1.7954 | 2.264 | 1595.3 | 0.037 | 712.9 | 79.8 | promoted |
| 763ac845601b | fxy_ood_T6T4_f109 | 109 | 1.9544 | 1.7632 | 2.143 | 1111.8 | 0.037 | 548.0 | 75.7 | promoted |

Note what the joint-clean gate costs and why it is still right: it excludes eight cores with
**lower** `F_xy` than 1.7836, including the global minimum 1.6926 — every one of them
excluded by `cbc_max` alone (1602–1663 ppm, i.e. 2–63 ppm over the 1600 gate), never by
`f_q` or AO. Those eight are not lost: the closest of them, the global minimum itself, is
carried in set B, so the wave still measures the record core.

### 3.2 `replicate_span20` (role `calibration`) — the replicate/determinism arm, §6 item 2

**Rule:** one core from **each of the 19 r1 strata** — the row nearest that stratum's own
`F_xy` median among its converged, labelled rows, disjoint from `fxywall_top20`, ties on
`record_id` — **plus** the global minimum-`F_xy` r1 core (`5cecf2d73b1b`, 1.6926). Span
`F_xy` **1.6926 – 2.4798**.

Why per-stratum medians rather than 20 more frontier cores: determinism is a property of the
*evaluation*, not of the objective, so the replicate set must sample where the evaluation
varies — across pair, feed, restart provenance and generator rule — not where the objective is
best. One core per stratum covers all 19 r1 cells, 5 feeds (109/117/121/125/129), all four
restart-provenance kinds, and spans `F_xy` 1.69–2.48, i.e. the middle 90% of r1's support.
The role tag `calibration` is `pinbu_analyze.py`'s existing taxonomy (`pinbu_analyze.py:190,
221`); this set is genuinely a predicted-pin **span** (predicted 69.4–94.2), so the existing
calibration-curve readout stays meaningful on it and no analysis code changes.

Full table: manifest `targets[]` where `group == "replicate_span20"`. Extremes:
`5cecf2d73b1b` (F_xy 1.6926, f125) … `b93ac9fee1c6` (F_xy 2.4798, `fxy_sf_e594_f117`).

### 3.3 Both sets are exact replays

Every one of the 40 chains is a replay of a stored pattern under the identity gate of §1, so
**all 40** contribute to the determinism readout and **all 40** produce a pin value. The A/B
split is what each set was *selected* for, not what it can report.

---

## 4. Deck

`pinbu_wave_fxyera_r1_199.inp` is a clone of `pinbu_wave_keep_199.inp` with three deltas and
no others (`diff` on the non-comment lines is in §9):

1. `[verify] harvest_maps = true` (was `false`)
2. `[model] cond_schema = "v8"` (was `"v7"`) — **inert** on the run path; set to the r1 deck's
   value so the deck reads as an r1-era artifact
3. header/title text

`[master] keep_success = true` is inherited from the `keep` deck and is **load-bearing**:
`F_xy` is MASTER's `FXYP`, printed **only in `MAS_OUT`**, and it is not on `WaveFom` — so
`pinbu_wave.py` cannot report it and the replay's `F_xy` must be read **offline** from the
retained final-cycle `MAS_OUT`, with the same scanner that produced
`fxy_backfill_199_produce_fxyera_r1_20260829.csv`. `keep_success=false` purges 100% of those
dirs (`pinbu_rodavg_20260820.md`). No retained `MAS_OUT`, no `F_xy` determinism arm.

`harvest_maps = true` does two things here, both wanted: `curriculum.py:1075` widens
`keep_success` to `keep_success OR harvest` (retention no longer rests on one knob), and
`_factory` installs `HarvestingEquilibriumEvaluator` so `WaveOutcome.maps` carries the final
EDIT5 map instead of `None` (`curriculum.py:1100-1104`). **Honest limit, stated so no one
looks for it later:** `pinbu_wave.py._outcome_record` does not persist `outcome.maps`, so this
wave writes **no** `maps.npz`. That costs nothing — 537 of the 540 labelled r1 rows already
carry a non-null `maps_key` — and the artifact that matters is the retained
`MAS_SUM`/`MAS_OUT` on disk.

`[verify] package_root = "FEASIBLE_PACKAGE"` is retained but **never resolved** by this wave
(no ga80 target); the launcher reports its absence rather than refusing.

**Run dir stays relative** (`--run-dir runs/pinbu_wave_fxyera_r1`). Nothing in the deck names a
drive, so if `runs/` is a junction to another volume the wave follows it with no edit; the
launcher and the status probe resolve `runs/`'s real target and report free space on **that**
volume, not on `C:`.

---

## 5. Registered marks (scored strictly; a failed control is a refusal, never a downgrade)

**Primary — the two things §6 asked for:**

- **M1 — pin axial peak on the `F_xy` frontier.** Reported: the **rate of measured
  `max_pin_burnup` ≤ 80.0 GWd/tU among the 20 low-`F_xy` `fxywall_top20` cores**, plus the
  ≤ 78.0 acquisition-gate rate and the full 20-core distribution. **Pre-registered
  prediction: ≥ 80% of the 20 measure ≤ 80.** Basis: s1i predicts 17/20 (85%) ≤ 80 and 9/20
  ≤ 78, and the 2026-08-20 wave measured the head's bias inside the 2.0 GWd/tU margin the
  decks already spend. **Falsifier:** < 80% measuring ≤ 80 means the `F_xy`-selected frontier
  is pin-limited, and pin becomes a *live* constraint on `F_xy` search rather than an advisory
  column — a decision-grade result either way.
- **M2 — `F_xy` determinism.** For every replay that reproduces its stored `f_r`/`cyclen`/
  `cbc_max` inside `TOL_F_R`/`TOL_CYCLEN`/`TOL_CBC` (0.002 / 0.5 / 2.0) **and** keeps its
  restart provenance, report `|F_xy_replay − F_xy_stored|` read from the retained `MAS_OUT`.
  **Pre-registered prediction: `|ΔF_xy| ≤ 0.002` on every such chain (max over 40 ≤ 0.002).**
  Basis: MASTER is deterministic on this fleet — the 2026-08-20 wave reproduced `f_r` to
  **0.0000** on 44/44, `cyclen` to 0.008, `cbc_max` to 0.01, `f_q` to 0.0001 — and `%EDT_OPT
  ipin` is an EDIT flag, output only. **Falsifier:** any chain exceeding 0.002 means the r1
  `f_xy` labels carry replay noise of that order, and every `F_xy` margin quoted against the
  1.65 limit must be widened by the measured spread before it is used in a siting decision.

**Secondary (measured and reported; failure does not invalidate the wave):**

- **M3 — chain integrity.** 40/40 converged, 40/40 `provenance_ok`. Provenance drift is the
  known failure mode: it dropped 3 of 44 chains on 2026-08-20. See §7.
- **M4 — pin-head skill on an `F_xy`-selected set.** `pin_error_pred_minus_meas` pooled and by
  group. This is the first test of the s1i pin head on cores selected by `F_xy` rather than
  `F_r`; the 2026-08-20 verdict was measured on an `F_r`-selected set.
- **M5 — scalar determinism, restated.** Max `|Δ|` over the 40 chains on `f_r`, `f_q`,
  `cbc_max`, `cyclen`, `ao_abs`, `max_assembly_burnup`, against the 2026-08-20 numbers above.
- **M6 — `F_xy`/`F_r` ratio on measured pairs.** Every chain yields a fresh (`F_xy`, `F_r`)
  pair, so r1 secondary `S1` (the wall-location prediction: ratio median in [1.04, 1.08] in
  the `F_r ∈ [1.50, 1.65]` region) gets a 40-point replication of its *stored* value. Pinned
  now, from the manifest: over these 40 targets the stored ratio runs **1.0240 – 1.1220,
  median 1.0758** — inside the registered interval but in its upper half. Over all 540
  labelled r1 rows: median **1.0839**, p25–p75 1.0703–1.1011 — **above** the registered
  interval. Caveat, stated so the replication is not over-read: **r1 produced no converged
  core with `F_r` in [1.50, 1.65]** (n = 0), so `S1` cannot be scored in the region it was
  registered for; these ratios are measured at higher `F_r` and are an extrapolation toward
  it, not a test of it.

---

## 6. Budget

| item | value |
|---|---|
| chains | 40 (20 + 20), one per core, no replication beyond the registered replicate set |
| workers | 12 (`[produce] workers`), all-P box, `use_all_cores = true`, `host_reserve = 0` |
| per-chain wall, precedent | median 359 s, max 630 s (44 chains, 12 workers, 2026-08-20); paramA subset median 362 s, max 548 s |
| wave wall, precedent | 34 ga80 chains in 1,686 s; 10 paramA chains in 548 s |
| **expected wall** | **~0.6 h** (40 paramA chains ≈ 40/34 × 1,686 s ≈ 33 min, plus staging) |
| **registered cap** | **1.6 h.** Exceeding it is reported, not silently absorbed |
| hard bound | `chain_timeout = 3600 s` × ⌈40/12⌉ = 4 batches |
| disk | 40 retained final-cycle dirs; launcher refuses below 15 GB free **on the volume `runs/` resolves to** |
| RAM | launcher refuses below 30 GB free (12 concurrent chains) |
| store writes | **none during the run** (`fuel_types.parquet` is read-only); merge is a separate `patch` step, dry-run first, backed up |

This spend was registered inside r1's budget (`produce_fxyera_r1_prereg_20260829.md` §5).

---

## 7. The one real risk: restart-provenance drift

18 of the 40 targets carry `restart_provenance` beginning `promoted:` — they resolve out of
the produce **promoted cache** (`data/produce/promoted`). If that cache has been pruned since
r1, the replay silently falls back to `pair_ecore:`/`pair_feed:`, `provenance_ok` goes
`False`, and `pinbu_wave.py patch` refuses the pin value (correctly: a different restart is a
different evaluation, and attaching its pin to the stored row would mix two chains). This is
exactly what dropped 3 of 44 chains on 2026-08-20.

Two mitigations, both already in place:

1. The launcher **refuses to start** if `data\produce\promoted` is missing, and prints the
   number of cached cells so a partial prune is visible before 0.6 h is spent.
2. The status probe reports `PROVENANCE_DRIFT` as a running count, so a drifting wave can be
   stopped early rather than read at the end.

Registered mix, for scoring: `promoted` 18, `native` 14, `pair_ecore` 4, `pair_feed` 4; 17
distinct restart files. A local dry-run on the coordinator resolves the `promoted:` ones to
`pair_ecore:` **because the coordinator kit has no promoted cache** — that is expected and is
not evidence about 199.

---

## 8. Hashes (pinned before launch; the launcher gates on every one)

| artifact | bytes | sha256 |
|---|---:|---|
| `pinbu_wave_fxyera_r1_199.inp` | 7,345 | `9DF97D687652F7BA5FCD17234629AAF82134BED57E8A6CB790EB621E7620FD4B` |
| `data/reports/pinbu_wave_fxyera_r1_manifest.json` | 90,857 | `53D139C2BA7A42207EF0BED9DF26B558A62B76750AD9E88797434434E6E00D91` |
| `pinbu_wave.py` | 36,222 | `5B3688CFAD684E9E837910F8842F68A2F2C21F931052DD52DE98262BA3581047` |
| `run_pinbu_wave_fxyera_r1_199.bat` | 1,841 | `BD51ED2DF4762AF85DFCD88E4F340005E174DDB961B37D9D0FF1D25F28A2D2CC` |
| `data/store/records.parquet` | 22,538,411 | `F38666E9F1508D35D33E0C22F583C5479C6F09CAC748201B494B47C8CFECA6EA` |

The store gate is a **sha256** gate, not the byte-length gate the 2026-08-20 launcher used:
the manifest pins 40 stored `f_xy`/`f_r`/`cyclen`/`cbc_max` labels and those labels *are* the
determinism reference, so a store that differs at all leaves the wave unscored. The value
above is the coordinator's **current local** store (75,793 rows). The harness sha is the
current `pinbu_wave.py`; it differs from the value the 2026-08-20 launcher gated on, so the
launcher must not be copied from that one without re-hashing.

---

## 9. Pre-launch validation already performed (coordinator, read-only, no MASTER)

| check | command | result |
|---|---|---|
| deck is a legal `LpoptConfig` and its assets | `python -m lpopt check --input pinbu_wave_fxyera_r1_199.inp` | `0 PASS, 2 FAIL, 1 SKIP` — **both FAILs are coordinator-box absences**: `master.executable` (`C:\DeCART_MASTER\BIN\master4.0m4_r1.exe`, lives on 199) and `verify.package_root` (`FEASIBLE_PACKAGE`, the ga80 kit dir, absent here and unused by this wave). `pinbu_wave_keep_199.inp` returns the **identical** two FAILs, so the result is host state, not deck state. |
| deck delta vs precedent | `diff` on non-comment lines, `pinbu_wave_keep_199.inp` → this deck | exactly the three deltas of §4 (`harvest_maps`, `cond_schema`, title/seed) |
| manifest schema | `schema == "pinbu_wave_prereg_v1"`, 40 targets, 40 unique `record_id`, 19 strata, all `library_id == "paramA"` | pass |
| **record_id identity, all 40** | `compute_record_id(pattern.canonical(), library_id, case_pair, PRODUCE_DECK_KNOBS)` vs stored | **0 drift** |
| **asset resolution, all 40** | `python pinbu_wave.py run --plan <manifest> --deck <deck> --run-dir <tmp> --dry-run` | 40 entries printed, **0 `[SKIP]`**, one paramA verifier, `workers=12` |
| launcher / probe syntax | PowerShell AST `Parser::ParseFile` | both `PARSE_OK` |

---

## 10. Execution order (to be run only on the coordinator's instruction)

```
# 1. ARM (the ONLY command that starts MASTER)
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_pinbu_wave_fxyera_r1_199.ps1"

# 2. WATCH (read-only, starts nothing)
ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_pinbu_wave_fxyera_r1_199.ps1"

# 3. MERGE the measured pin column — dry run FIRST, always
python pinbu_wave.py patch --results runs/pinbu_wave_fxyera_r1/pinbu_wave_results.jsonl --dry-run
python pinbu_wave.py patch --results runs/pinbu_wave_fxyera_r1/pinbu_wave_results.jsonl --tag pinbu_fxyera_r1_20260830
```

A partial run **resumes**: re-invoke the `.bat`, never the launcher (the launcher deletes the
run dir, and the results JSONL in it is the only resume ledger).

The `F_xy` determinism arm (M2) is **not** produced by any of the three commands above: it is
an offline pass over `runs/pinbu_wave_fxyera_r1/paramA/master_work/**/MAS_OUT*` with the r1
`FXYP` scanner, joined to `stored.f_xy` in the manifest on `record_id`. That readout, and the
results report `data/reports/pinbu_wave_fxyera_r1_results_<date>.md`, are the next
deliverable — not part of this one.

**Deliverables of this pre-registration, all present and hashed, nothing launched:** deck,
manifest, launcher trio (`launch_` / `run_` / `status_pinbu_wave_fxyera_r1_199`), this
document.
