# Round 13 — ADDENDUM: arm S1i (`cond_schema = v8`, the 5-type scorer)

**Written 2026-08-17, BEFORE `data/splits/S1i.json` was persisted and before the
arm was launched.** Same discipline as `ab2_addendum_S1H_20260817.md` (round 12);
instrument, split rule, preservation checks, recipe freeze and noise-floor
reporting rules inherited unchanged. Only what is new is stated here.

**Instrument = gate-promote** against champion **`s1h`** (`gate_s1h.json`:
`pass: true`, legacy-tail `pass: true`, ε 2.0). **The gate is NON-REGRESSION and
gains are not expected** — see §1.

---

## 1. Why this arm exists — it is a capability change, not an accuracy bid

`cond_v8` widens v7's fresh-type composition block from width 3 to width 5. Its
**purpose is to make 4- and 5-type cores SCORABLE** for the multi-type mesh sweep
(P3). It is not a bid to predict better on the 2-type corpus that dominates
training, and this addendum does not ask the gate to show that it does.

> **Registered expectation: no gain. The bar this arm must clear is
> non-regression.** A PASS means "the wider scorer costs nothing"; it does not
> mean "the wider scorer helps".

---

## 2. Frozen inputs

| item | sha256 | bytes |
|---|---|---:|
| `data/store/records.parquet` | `b55e108d7d793b190878cffcbd770abdca24e93746ff0a405aeeee8bdf5f57bc` | 22,170,018 |
| `data/store/maps.npz` | `4215c94d6a51d2dadbdd23e17a43822ae4ea506896dbe18031f6881d16ffb179` | 209,818,497 |
| `data/store/fuel_types.parquet` | `fc73ad29741815612c86d91df746258d20bf9513652a93ea388924b081f78137` | 64,343 |
| parent `data/splits/S1h.json` | `5dfb6b05f05d698a82881d26db7d2deb0fd9639d6789e64cd24e9030ee31947a` | 6,036,074 |

Rows **74,537** · increment vs S1h = **60**, campaign `fpcamp_minfr_triple_f125`,
feed 125, `case_pair P6253Z1G06N24_P6253Z2G10N20_P6253Z2G10N24` — **the
programme's first 3-type labels**, 49 converged
(`tripletype_f125_results_20260817.md`). `fuel_types.parquet` unchanged for the
sixth round running. **Control:** `data/models/s1h`, frozen on disk.

**Recipe frozen** per `scaling_results_20260815.md`: `224 / 8 / 384`, **5 members
not 10**, v6b recipe verbatim; the *only* change from `s1h` is
`--cond-schema v8` (was v7) plus `--split S1i`. Noise floors bind reporting
(round 10 §2.2): movement below **~0.01 on cyclen / map_cov is not
attributable**; single-seed spread sd ≈ 0.018 on ρ F_r; `mde80` ≈ 0.009–0.015.

---

## 3. What v8 actually changes — measured, not assumed

| | v7 | **v8** |
|---|---|---|
| cells | 62 | **62 — bit-identical tuple** |
| `prior_power` index | 50 | **50** |
| globals | 18 | **20** |
| added | — | `g_type_frac_4`, `g_type_frac_5` |

### 3.1 v8 is NOT append-only in the globals vector — registered

Every schema step since v6 has been append-only, and **this one is not**. v8
*inserts* the two new fractions **before** `g_e_type_std` / `g_n_fresh_types`:

```
v7 : … g_type_frac_1, g_type_frac_2, g_type_frac_3,                         g_e_type_std, g_n_fresh_types   (18)
v8 : … g_type_frac_1, g_type_frac_2, g_type_frac_3, g_type_frac_4, g_type_frac_5, g_e_type_std, g_n_fresh_types   (20)
```

Verified: `v8.globals[:18] != v7.globals` — **prefix-stability does not hold.**
That is deliberate (the composition block is a contiguous width-5 field) but it
has consequences that must be stated:

* a v7 checkpoint and a v8 checkpoint are **not** mutually loadable, and the two
  cannot be compared without re-encoding;
* positional weights for `g_e_type_std` / `g_n_fresh_types` do **not** carry over
  from v7 — harmless here only because this arm trains **from scratch**.

The **cells** are bit-identical, so nothing in the convolutional path moves.

### 3.2 The composition globals' degeneracy — what the 49 rows do and do not fix

Measured across the whole store by `case_pair` token count:

| fresh types per core | rows |
|---|---:|
| 2 | **74,477** |
| 3 | **60** |
| 4 | **0** |
| 5 | **0** |

Two distinct facts, and round 12's note is about the first of them only:

1. **`g_type_frac_3` stops being ≡-degenerate.** At `s1h` train time every core
   was 2-type, so that slot was identically zero — the degeneracy `s1h`
   documented. The 60 triple rows (49 converged) give it its first non-zero
   values (sampled 0.416–0.480). **60 rows is 0.08% of the corpus; it breaks the
   degeneracy without teaching much**, and this addendum claims nothing more.
2. **`g_type_frac_4` and `g_type_frac_5` remain identically ZERO in every
   training row** — there are no 4- or 5-type cores anywhere in the store.
   The model can learn *nothing* for them. They are **inert at train time** and
   exist solely so the input vector is the right width to *score* 4/5-type
   candidates at serve time.

> Registered consequence: **v8's own two new globals contribute no gradient
> signal in this round.** Any measured v8-vs-v7 difference must come from the
> re-encoding, the 60 new rows, or seed/run noise — *not* from the new globals
> having been learned. A post-hoc story attributing a gain to `g_type_frac_4/5`
> is excluded in advance.

### 3.3 The confound, and why it is cleaner than round 12's

Round 12 §4.1 registered a **three-way** confound (schema + recipe + data). This
arm has **two** differences from its control, both small and both named:

| | s1h (control) | S1i |
|---|---|---|
| schema | v7 (18 globals) | **v8 (20 globals)** — cells identical, globals widened |
| data | 74,477 rows | **74,537** (+60) |
| recipe / architecture / seeds | v6b, 224/8/384×5, 20260716–20 | **identical** |

So `S1i − s1h` = **widening + 60 rows**, with the widening's own new fields
provably inert (§3.2). It is a materially cleaner comparison than round 12's, and
that is stated as an improvement in attribution, not as evidence of anything.

---

## 4. The split — S1i

`build_split_S1b.py --parent S1h --name S1i --holdout-new-campaigns`. S1h
assignments verbatim; stable-hash 80/20 on the increment only. One new-only
campaign.

| | |
|---|---:|
| train / val | **62,429 / 12,108** (+50 / +10) |
| cells created | 1 (`fpcamp_minfr_triple_f125`) · grown: 0 |

| # | check | result |
|---|---|:--:|
| **a** | every S1h **val** id stays val | ✅ |
| **b** | every S1h **train** id stays train | ✅ |
| **c** | only the 60 new ids get fresh assignments | ✅ |
| **d** | every S1h cell present, ids retained, order intact | ✅ |
| **d′** | **`ab2_frozen_val_by_cell` forwarded verbatim — 3,207 rows** | ✅ |
| — | no train/val overlap · every store row in exactly one fold | ✅ |

**Cost of the uniform rule, disclosed:** the 80/20 holdout sends **10 of the 49
converged triple rows to val**, so only **39** remain to train the sole
non-degenerate composition signal. Applying the rule uniformly was chosen over a
special case — with 49 rows the block is effectively unlearnable either way
(§3.2), the gate is non-regression, and a bespoke exemption would be a fourth
rule variant for no measurable gain. Round 10 §3.1's interventional-wave
exclusion continues to apply to all reported readouts.

---

## 5. Decision rule — fixed before the gate runs

### 5.1 Primary

**`lpopt gate-promote`**, prev = `data/models/s1h`, new = the pulled candidate,
both on **S1i**. Promotion **authorized on PASS**; the gate JSON is written to
`data/reports/gate_s1i.json` before promotion is acted on. Generalization is
judged on the **frozen 36-cell surface** (3,207 rows), which contains none of the
triple rows.

Per round 12 §5.1, each model is served **under its own schema** — `s1h` at v7,
the candidate at v8 — on the identical row set and order; the schemas are not
interchangeable (§3.1), so this is required, not optional.

### 5.2 Registered SECONDARY readouts — decide nothing

**(a) The triple cell `fpcamp_minfr_triple_f125`** — 10 val rows. Below
`ab2_paired.MIN_CELLS = 3` as a cell count, and 10 rows is far below any
meaningful interval: reported as a **descriptive** MAE/Spearman with **no
interval**, and it is the champion's first exposure to 3-type cores at serve time.

**(b) The standing T6_T4 series** — unchanged by construction (no T6_T4 rows in
this increment), so it functions as a **stability check**: movement there would
indicate collateral drift from the widening, not progress.

### 5.3 Falsification — registered before the gate runs

> If gate-promote FAILS, S1i is rejected and `s1h` stands, and the registered
> reading is that **the width-5 widening costs accuracy on the 2-type corpus**.
> Because §3.2 proves the two new globals are inert at train time, a FAIL could
> not be attributed to them being "wrong"; it would have to come from the
> re-encoding of `g_e_type_std` / `g_n_fresh_types` into new positions or from
> the 60 rows. The registered next step on a FAIL is therefore to re-run v8 with
> the composition block **appended** rather than inserted — isolating position
> change from width change — before abandoning 4/5-type scoring.
>
> **A FAIL would also block P3**: without a promoted v8, the multi-type mesh
> sweep has no scorer for 4/5-type candidates.

### 5.4 What this round cannot conclude

No same-recipe-same-data control. A PASS promotes a model; it does **not**
establish that the widening or the 60 rows helped — and §1 registers that no gain
is expected in the first place. Per §2, no delta below ~0.01 on cyclen/map_cov may
be attributed to this arm at all.

---

## 6. Pull, scoring and promotion

```
python -m lpopt.remote --input lpopt.inp pull --ts s1i        # -> data/models/s1i/

python -m lpopt gate-promote --input lpopt.inp \
    --prev data/models/s1h --new data/models/s1i \
    --out data/reports/gate_s1i.json --check-only
# promotion AUTHORIZED on PASS (§5.1): re-run without --check-only
```
