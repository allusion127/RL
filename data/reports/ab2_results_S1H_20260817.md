# Round 12 — RESULTS: arm S1h (`cond_schema = v7`, the 3-fresh-type scorer)

Scored against `data/reports/ab2_addendum_S1H_20260817.md`, written before the
split was consumed and before the arm was launched. Every readout registered
there is answered below, including the two that are unflattering.

---

## 1. Verdict — PASS, promoted, and the programme can now score a triple

| | |
|---|---|
| training | 238 GPU 1, `runs/s1h`, `rc=0`, **DONE**. Featurize 1,154 s; 5 members in one chunk, 9,353 s; all five early-stopped (epochs 48–108). |
| checkpoint | `cond_schema=v7`, 5 members, 62 cell channels / **18 globals** |
| gate | `data/reports/gate_s1h.json` — **PASS** |
| no-regression | `worst_drop 0.0214` vs `epsilon 0.1388`, `blind_targets []` |
| legacy tail | PASS (`epsilon 2.0`) |
| promotion | executed — `data/curriculum/state.json` and `lpopt.inp` both now read `data/models/s1h` |

`s1h` is the **9th champion** and the first checkpoint in the programme that can
encode a 3-fresh-type core. That, not the gate delta, is what the round was for.

**Recorded because the gate's own console said it and a bare "PASS" would hide
it:** `f_r` is a **REPORT-ONLY** axis in this gate — it was scored, not enforced
(`worst 0.0134` vs the same epsilon). A PASS does not mean `f_r` was verified
regression-free. Unchanged from every prior round; `[curriculum]
gate_noreg_fr_guard_enabled` is still false.

---

## 2. §5.2(a) — the T6_T4 stability check (the informative one)

Registered as the round's most informative secondary *because its data did not
change*: no T6_T4 row is in the increment, so any movement is collateral drift
from the schema swap, not progress. **183 val rows, 8 cells** (7 with ≥ 8 rows),
interventional waves excluded per round 10 §3.1.

| axis | s1g (champion) | s1h (candidate) | Δ | attributable? |
|---|---:|---:|---:|---|
| ρ F_r | 0.8383 | 0.8555 | **+0.0172** | **NO** — single-seed spread on ρ F_r is sd ≈ 0.018 |
| MAE F_r | 0.0319 | 0.0311 | −0.0008 | no |
| ρ cyclen | 0.9561 | 0.9529 | −0.0032 | **NO** — below the ~0.009 ensemble-identity floor |
| MAE cyclen | 0.6860 | 0.6239 | −0.0621 | — |
| ρ CBC | 0.9445 | 0.9499 | +0.0054 | no |

**Reading: no collateral drift.** Every movement sits inside the noise floors
§2 of the addendum bound the round to. That is the result this readout was
registered to produce, and it is the one it produced.

## 3. §5.1 — the frozen 36-cell surface (3,207 rows), which the gate judges on

| axis | s1g | s1h | |
|---|---:|---:|---|
| ρ F_r | 0.9560 | **0.9589** | + |
| MAE F_r | 0.1117 | **0.1049** | improved |
| ρ cyclen | 0.9988 | 0.9989 | flat |
| MAE cyclen | 2.9816 | **2.8563** | improved |
| ρ CBC | 0.9975 | 0.9978 | flat |
| MAE CBC | 24.5787 | **22.3817** | improved |

Consistent with the gate. None of these rows is in the increment.

## 4. §5.2(b) — the three new frontier cells (fit checks, near-tautological)

The champion holds **zero** of these rows; the candidate trains on ~80 % of each.
A gain here confirms absorption and nothing else.

| cell | n | MAE F_r s1g → s1h | ρ F_r s1g → s1h |
|---|---:|---|---|
| `fpcamp_minfr_E1E2_f109` | 20 | 0.0481 → **0.0283** | 0.8376 → 0.8180 |
| `fpcamp_minfr_hgd569_f125` | 11 | 0.0394 → **0.0243** | 0.6727 → 0.6091 |
| `fpcamp_minfr_hgd569_f109` | 8 | 0.0709 → **0.0427** | 0.3095 → 0.4524 |

**MAE improves in all three; Spearman wanders.** At n = 8–20 the rank statistic
is not a measurement — `hgd569_f125`'s cyclen ρ moves 0.7727 → 0.2364 while its
cyclen **MAE improves 17.74 → 5.10**, which is exactly what a rank statistic does
on 11 points once the level error collapses. Three cells is `MIN_CELLS = 3` and
below `MIN_CELLS_BCA = 6`, so no interval is quoted, as registered.

## 5. §5.2(c) — the eleven `mv3_*` anchor strata (breadth)

34 val rows / 11 cells. MAE F_r 0.1004 → 0.0988; **cyclen MAE 4.88 → 2.76**;
**CBC MAE 53.68 → 29.12**. Large gains, expected — these rows are now in train.
Their F_r spans 1.82–3.69, entirely outside the F_r < 1.55 decision band, so per
the addendum this is breadth coverage and **never** evidence of search skill.

## 6. §5.2(d) — `mesh_vs_db.py` NOT run, as registered

Dropped in advance because a v6b-vs-v7 frontier-bias comparison inherits the §4.1
confound with no way to attribute it. **`data/reports/scoping_mesh_20260815/`
was not touched** — the s1f/s1g baselines are intact.

---

## 7. UNREGISTERED FINDING — cross-cell calibration got modestly worse

This was not a registered readout. It is reported because it is real, it is not
visible in anything that was registered, and burying it would be the dishonest
choice.

Pooled over all 12,035 val rows and all 89 cells at once:

| pooled axis | s1g | s1h | |
|---|---:|---:|---|
| ρ F_r | 0.2716 | **0.2128** | worse |
| MAE F_r | 0.5351 | **0.5568** | worse |
| ρ cyclen | 0.4824 | **0.3924** | worse |
| MAE cyclen | 14.48 | **21.45** | worse |

That looks alarming next to §2–§5, and the diagnosis resolves it: it is
**Simpson's paradox**, and the mechanism is measurable.

| | s1g | s1h |
|---|---:|---:|
| **within-cell** mean ρ F_r (62 cells ≥ 8 rows) | 0.8621 | 0.8605 (Δ −0.0016, 35/62 cells improved) |
| **within-cell** mean ρ cyclen | 0.8525 | 0.8594 (Δ +0.0069, 44/62 improved) |
| **cross-cell** per-cell bias sd, F_r | 0.1301 | **0.1330** |
| **cross-cell** per-cell bias sd, cyclen | 4.8752 | **5.8757 (+20 %)** |

**Within-cell ranking — the axis the gate measures, and the axis the search
uses — is a wash** (both deltas below their noise floors). What degraded is the
**level the model places each cell at relative to the others**: the spread of
per-cell mean prediction error widened, most clearly on cyclen. Pooling then
converts that level spread into an apparent ranking collapse.

**What this does and does not mean.**

* It does **not** overturn the promotion. The registered instrument is the
  per-cell no-regression gate; it passed, and §2–§4 confirm no within-cell
  regression on any surface.
* It **does** flag a cost that no registered readout would have caught. Cross-cell
  level is what the trust-region support logic and any cell-to-cell comparison
  read. A +20 % widening on cyclen bias sd is small in absolute terms
  (4.88 → 5.88 EFPD) but it is a *direction*, and it is the second round in a row
  that the schema carrier (v6c's ADF block) has failed to pay for itself.
* Attribution is **impossible** in this arm, exactly as §4.1 pre-registered:
  schema, ADF block and +474 rows all moved together.

**Registered next step, written now so it is not chosen after the fact:** if the
next round wants this isolated, the run to make is the one §5.3 already named —
`--cond-schema v7` on the **parent split S1g**, data held fixed. That single arm
separates schema+ADF from data growth on exactly this axis.

---

## 8. What this round did not establish

Verbatim from the addendum, and all of it still holds:

1. **No same-recipe-same-data control.** A PASS promotes a model; it does not
   establish that data growth is the lever.
2. **Three changes, one arm** (§4.1): +4 ADF cell channels, +5 composition
   globals, +401 train rows. No delta is attributable to any one of them.
3. **The composition globals could not have been learned** (§4.2). On a corpus
   where every row is 2-type, `g_type_frac_1 ≡ g_split_frac` exactly,
   `g_type_frac_3 ≡ 0` and `g_n_fresh_types ≡ 2/3` are constant, and
   `g_e_type_std ≡ sqrt(w(1−w))·g_e_split` to 6e−08. v7 bought
   **representational capacity, not knowledge**. Its first 3-type input is
   out-of-distribution on channels whose weights were fitted against constants.
4. No delta below ~0.01 on cyclen / map_cov may be attributed to this refresh.
5. `f_r` was scored but **not enforced** by the gate (§1).

---

## 9. Provenance

| item | value |
|---|---|
| split | `data/splits/S1h.json` sha256 `5dfb6b05f05d698a82881d26db7d2deb0fd9639d6789e64cd24e9030ee31947a` — 62,379 / 12,098 (+401 / +73), 14 cells created, 0 grown |
| store | `records.parquet` sha256 `b5b29460…cae7`, 74,477 rows (65,922 converged) |
| gate JSONs | `gate_s1h_checkonly.json` (inspection), `gate_s1h.json` (promoting run) |
| scored rows | `runs/s1h/rows_s1h_cand.csv`, `runs/s1h/rows_s1h_champ.csv` — 12,035 rows, identical ids and order, 89 cells |
| deck | `lpopt.inp` `model_dir` → `data/models/s1h`, `cond_schema` → `v7` (hand-updated: `gate-promote` rewrites `model_dir` only; `tests/test_config.py::test_cond_schema_default_is_v3` is the tripwire that caught the stale value) |
