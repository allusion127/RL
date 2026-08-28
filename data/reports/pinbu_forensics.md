# max_pin_burnup ranking-failure forensics

Champion under audit: `data/models/20260718_200838` (cond_v4, 43ch). All numbers
below are read-only analysis over `data/store/records.parquet`
(41,163 rows: A=38,854, P=1,735, B=574) and the champion scored on CPU.

## TL;DR verdict

1. **The reported "-0.818 actively anti-correlated" overstates the problem — it
   is a small-sample (n=11) artifact.** The better-powered held-out estimate for
   the same cell 5 (n=30) is **Spearman -0.06** (bootstrap 95% CI [-0.42, +0.29],
   36% of resamples > 0). The n=11 blind (-0.818) and post-train (-0.373) numbers
   are the noisiest tail draws of a true out-of-sample skill of ~0.
2. **But `max_pin_burnup` is genuinely the only target that fails to generalize
   within-cell out-of-sample** — a real, structural weakness, not a bug in one
   cell. Across all 5 cells its held-out within-cell Spearman is 0.0–0.6 while the
   other five targets are 0.65–0.95.
3. **Mechanism = prediction collapse from two compounding causes:** (a) a
   label-fidelity/definition inconsistency between Dataset A and P, and (b) the
   axis being a single-assembly residence/local-peaking discharge *extreme* — the
   least-BOC-determined of the 7 targets — with only ~119 in-band training rows
   per cell, so the head memorizes (in-sample 0.99) instead of generalizing.
4. **Feature fix (per-type BRP channel) cannot help the within-cell failure**
   (proof below): within a fixed cell the fuel pair is fixed, so any per-type
   feature is constant across the ranked patterns and adds zero ranking signal.
5. **Implemented (d): demote `max_pin_burnup` to advisory** (reported, not gated).
   **Recommended (not implemented, needs GPU validation): (b)** censor Dataset-A
   pin labels and/or promote `max_assembly_burnup` to a first-class target.

---

## A1.1 — Label audit: where does `max_pin_burnup` come from?

### Extraction chain
- **Definition:** the maximum 3-D pin burnup *within the single assembly that has
  the maximum EOC assembly-average burnup*. Parser
  `lpopt/vendor/masterrl/burnup.py::parse_ppi_max_pin_burnup` reads the
  `PIN 3-D BURNUP DISTRIBUTION` block of `MAS_PPI` for one assembly; the target
  assembly is `metrics.max_burnup_assembly` (the SUMMARY EDIT5 peak
  assembly-average location), selected in
  `lpopt/vendor/masterrl/master.py:818-850`. It is the equilibrium-cycle EOL
  value = the lead assembly's peak-pin **discharge** burnup (the licensing quantity).
  **E17 확증 (2026-08-20)**: 이 "licensing quantity" 단정은 이제 **옳다** — 사용자
  판정으로 공식 관측량이 **핀 axial peak**(= 이 3-D 노드 첨두), 한도가
  **80 GWd/tU** 로 확정되었다 (`data/reports/pinbu_definition_20260820.md`). 봉평균은 보조 관측량이다.
  단 §1.3 의 **단일 집합체 한정**(EDIT5 최대 집합체 1기만 스캔)은 여전히 과소평가
  방향의 결함으로 남는다.
- It is a **real pin-resolved MAS_PPI value only for Dataset P** (produced with
  `enable_pin_burnup=True`, which turns on the `%EDT_OPT ipin=1` PPI output —
  `burnup.py::enable_ppi_output`, `design/coredeck.py`).

### Consistency across datasets A / B / P — **NOT consistent**
| dataset | rows | `max_pin_burnup` non-null | pin/assembly ratio | pin scale (GWd) | feed |
|---|---|---|---|---|---|
| A (MOCHA SA/GA cache) | 38,854 | **100%** | **1.080 ± 0.013** (CV 1.2%) | 65–81, mean 70.6 | **121 only** |
| B (3_GA) | 574 | **0%** (all NaN, censored) | — | — | — |
| P (produced, real PPI) | 1,735 | 52% (899) | **1.181 ± 0.031** (CV 2.7%) | 70–111, mean 85.2 | 109/117/125 |

**Finding — Dataset A and Dataset P `max_pin_burnup` are different physical
quantities.** Dataset A carries a pin value for *every* row despite being the
MOCHA SA/GA cache (which produces no MAS_PPI); its pin/assembly ratio is
near-constant at 1.080 ± 1.3% — the fingerprint of a lattice pin-peaking *factor*
applied to assembly burnup, i.e. it is essentially `1.08 × max_assembly_burnup`,
with almost no within-assembly peaking variance. Dataset P is the true MAS_PPI 3-D
peak: ratio 1.181 ± 2.7%, a *different definition, ~9% higher peaking, 2.3× more
variance, and a disjoint feed grid*. Dataset A supplies 99.9% of the non-null pin
labels (43:1 over P) but is a poor teacher for the P regime that is actually
evaluated.

### Is there variance to rank within a cell? — **Yes**
Cell 5 = campaign `5.5-5.75_f117`, Dataset P: n=149, mean 89.9, **std 6.4,
range 75.1–110.9 GWd**. The target is far from constant, so "near-constant →
Spearman noise" is **ruled out** as the cause. The failure is genuine model skill.

---

## A1.2 — Residual analysis with the champion

### In-sample vs out-of-sample (cell 5, split = the trained `data/splits/S1.json`)
| subset | n | Spearman(pred, act) | pred std | act std |
|---|---|---|---|---|
| ALL cell-5 rows | 149 | +0.855 | 4.97 | 6.42 |
| **IN-SAMPLE (train)** | 119 | **+0.990** | 5.45 | 6.70 |
| **HELD-OUT (val)** | 30 | **-0.061** | **2.21** | 5.14 |

The +0.855 on "ALL" is inflated by the 119 in-sample rows. On genuinely held-out
rows the model has **no within-cell ranking skill** and its **prediction spread
collapses** (std 2.21 vs actual 5.14) — it regresses to the mean for unfamiliar
patterns. Bootstrap 95% CI on the held-out Spearman: **[-0.42, +0.29]**.

### Reproducing the reported -0.818 (n=11 blind probe)
Predicted pin range [82.9, 86.8] (std **1.18**) vs actual [83.6, 98.8] (std
**5.07**): the model squashes all 11 candidates into an 84–86 band, so their rank
is noise and happens to be anti-aligned → Spearman -0.818. Same 11 rows,
post-train champion → -0.373. On the same n=11 subset **cyclen (0.29) and ao_abs
(0.35) are also weak**, confirming the subset — not a pin-specific pathology — is
the dominant factor.

### Multi-cell, well-powered held-out within-cell Spearman (n=30–60)
| cell | f_r | f_q | cbc_max | cyclen | ao_abs | **max_pin_burnup** |
|---|---|---|---|---|---|---|
| 5-5.25_f117 | +0.85 | +0.86 | +0.90 | +0.65 | +0.77 | **+0.38** |
| 5.25-5.5_f109 | +0.94 | +0.93 | +0.83 | +0.86 | +0.79 | **-0.03** |
| 5.25-5.5_f117 | +0.92 | +0.92 | +0.90 | +0.92 | +0.92 | **+0.40** |
| 5.25-5.5_f125 | +0.80 | +0.83 | +0.94 | +0.83 | +0.89 | **+0.64** |
| 5.5-5.75_f117 | +0.95 | +0.94 | +0.86 | +0.91 | +0.73 | **-0.06** |

`max_pin_burnup` is the **only** target that fails to generalize within-cell; it
never reaches the others' band and is the only one that goes negative.

### Confounder check
The anti-correlation is **not** "the model ranks by discharge/feed instead":
`spearman(pred_pin, pred_discharge)` is ~0 on the held-out sets. It is **variance
collapse** — the model has no discriminative spread out-of-sample. On the full
(in-sample-contaminated) population the prediction does track assembly burnup
(0.78–0.84), but that signal does not survive to held-out rows.

---

## A1.3 — Feature-support hypothesis

**Within a fixed cell the fuel pair is fixed** (cell 5 pairs = H1_H2, H3_H4,
G3_G4). Therefore the within-cell variation in `max_pin_burnup` comes *entirely*
from the loading pattern — the residence chain / shuffle path of the peak
assembly and where fresh assemblies land — which the CNN already sees via the
pattern maps.

**Consequence for candidate (a):** a *per-type* BRP-at-EOL origin channel (peak
pin BU / assembly-avg BU per fuel type) is **constant across the ranked patterns
within a cell** and so contributes **zero within-cell ranking signal**. It cannot
fix the reported failure; at best it recalibrates cross-cell scale, which is not
the within-cell-rank problem. Candidate (a) is therefore **not supported**.

**What actually drives within-cell `max_pin_burnup`:** it is 93% rank-correlated
with the peak assembly's EOL assembly-average burnup (`spearman(act_pin,
act_assembly)` = 0.90–0.94 across cells) — a single-assembly, multi-cycle
residence/discharge extreme. This is the least-BOC-state-determined of the 7
targets. The model does not even predict `max_assembly_burnup` (surrogate column 5
is left NaN by design; `discharge_burnup` is the core *average* and is NaN for all
Dataset-P rows).

---

## A2 — What was implemented, and why

**Chosen fix: (d) demote `max_pin_burnup` from a gate-driving target to advisory
(reported, not gated).** Justification from the evidence above:
- Its within-cell out-of-sample rank skill is ~0 and, at the n~11 probe holdout,
  its Spearman is noise-dominated (CI spans zero). Including it in the new-cell
  gate's `mean_spearman` injects noise; the pass threshold is 0.0, so a spurious
  negative can **false-fail an otherwise-healthy cell**.
- It is already report-only in acquisition (`criteria.pin_bu_limit` default
  `None` — `search/acquisition.py:747`), so this makes the gate consistent with
  the axis's existing advisory status.
- It touches no model/architecture/channel, so the deployed 43ch v4 champion and
  its parity guard are untouched; it writes no data/store/curriculum state.

Changes:
- `lpopt/config.py` — new `CurriculumConfig.gate_advisory_targets: list[str]`
  (default `["max_pin_burnup"]`).
- `lpopt/curriculum.py::_gate_newcell` — advisory targets are excluded from the
  `mean_spearman` aggregate but still reported per-target (each per-target row now
  carries `"advisory": bool`).
- `tests/test_curriculum.py::test_advisory_pin_burnup_excluded_from_newcell_gate`
  — a candidate that is perfect on the five gate targets and perfectly
  anti-correlated (-1.0) on `max_pin_burnup` still passes with `mean_spearman`
  == 1.0, and the pin row is flagged advisory with its -1.0 reported.

### Why NOT the other menu options
- **(a) per-type BRP channel / v5 cond schema** — proven inert within-cell (fuel
  fixed ⇒ constant feature). Not implemented.
- **(c) rank-normalize within-cell at loss time** — a scale/normalization fix; the
  failure is a within-cell *rank* (discrimination) failure, which a target
  rescale does not address. Not supported.
- **(b) fix/censor the label** — the label inconsistency is real (see A1.1) and is
  a plausible driver of the OOS collapse, but the fix (censor Dataset-A pin labels
  so the head trains on the 899 fidelity-consistent P labels) requires a GPU
  retrain to validate and could hurt cross-cell calibration (A provides the
  assembly→pin backbone). It is **recommended as the next experiment**, not
  implemented blind while a MASTER production is running.

## Recommendations (need GPU validation before adoption)
1. **Experiment: censor Dataset-A `max_pin_burnup`** at training time (a per-target
   mask in `dataset_torch.LPDataset._targets`, analogous to the `cbc_max`
   `boc_only` rule), keeping A/B for the other six targets. Measure held-out
   within-cell Spearman before/after. Expected effect: removes the
   "peaking≈constant" prior that causes the OOS collapse.
2. **If a licensing-grade within-cell pin RANKING is required**, add
   `max_assembly_burnup` as a first-class TARGET (already stored for P; it is the
   BOC/shuffle-determined quantity that drives 93% of pin rank and should
   generalize far better than the raw pin extreme), then map pin as a residual on
   top. Schema-additive, new head, validate on GPU.
3. **Do not pursue a per-type BRP origin channel for this problem** — proven inert
   within-cell.
4. Treat the n~11 probe Spearman as **underpowered** in all reports; prefer the
   n=30 per-cell held-out estimate for `max_pin_burnup`.
