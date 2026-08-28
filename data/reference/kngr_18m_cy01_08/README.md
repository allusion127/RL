# KNGR 18-month Multicycle Analysis — Extracted Reference Data

## Source document

- **File**: `KNGR_18month_CY01-08.pdf` (91 PDF pages), located at
  `...\99_Archive\KNGR_DOC\DOC\` (read-only source; not modified).
- **Identity**: KNFC Calc. Note **N-411-FN-D301-011 Rev.01**, "KNGR 18-month Multicycle
  Analysis", Korea Nuclear Fuel Co. Fuel Technology Center, prepared 02/26/99 (Rev.00
  03/20/98).
- **Scope**: KNGR (= APR1400 precursor design) 2-batch, 18-month multicycle core-follow
  analysis, cycles 1 through 8, using ROCS 3-D nodal depletion. Establishes reference core
  loading patterns and physics parameters (peaking factors, boron, MTC, etc.) later used for
  fuel performance / mechanical design inputs.
- **Page-numbering quirk**: the cover declares "Calc. Note 84 Pages", and the running footer
  on every body page reads "Page N of 84" — but the PDF has 91 physical pages. Pages 85-89
  (Tables 8.1-8.3, the Results detail tables) all print as "Page 85 of 84" ... "Page 89 of 84",
  i.e. they were appended after the original 84-page document was finalized without updating
  the "of 84" denominator. As a result, "9. REFERENCES" — which the Table-of-Contents (p.4)
  lists as starting on page 84 — actually appears on physical page 90, and the Korean-language
  Attachment A (independent-review checklist) is on page 91. This is a source-document
  artifact, not an extraction error.

## What was extracted

All extraction was done with PyMuPDF (`fitz`), text-layer only (no OCR needed — the
document's text extracts cleanly except where noted in Gaps below).

### 1. Per-cycle loading/shuffle maps → `pattern_cy01.csv` … `pattern_cy08.csv`

Source: Figures 6.2-6.9 "Loading Pattern for KNGR Cycle N", PDF pages 66-73 (one octant/
quarter-core map per cycle, legend `x = current cycle location, y = previous cycle location,
z = assembly type, s = rotation index`).

Columns: `position` (x), `prev_position` (y, blank if none printed), `type` (z). The rotation
index `s` was read (to correctly delimit cells) but is **not** included in the CSV, per the
requested column set.

**Extraction method**: word-level extraction with coordinates (`page.get_text("words")`),
not plain `get_text()`. Each grid cell in the source layout is a small stack of PDF text
blocks (`x [y]` on one line, `z` centered ~13pt right and ~20pt below, optional `s` directly
below `x`). A parser walks the coordinate-sorted word stream keeping an "expected next
position" counter (positions in every cycle's diagram are numbered sequentially 1..69 in
raster order) and disambiguates the optional trailing rotation digit from the next cell's
position label purely by its vertical offset (Δy ≈ 36pt = rotation row; Δy ≈ 0 or 50pt =
next position), never by its numeric value — this avoids the value-collision trap where a
rotation index (0-3) coincidentally equals the next position number.

**Per-cycle verification (checksums)**:

| Cycle | positions parsed | ambiguous (`?`) | skipped/stray tokens | fresh (blank-y) cells | shuffled cells |
|---|---|---|---|---|---|
| 1 | 69/69 | 0 | 0 | 69 | 0 |
| 2 | 69/69 | 0 | 0 | 22 | 47 |
| 3 | 69/69 | 0 | 0 | 26 | 43 |
| 4 | 69/69 | 0 | 0 | 26 | 43 |
| 5 | 69/69 | 0 | 0 | 27 | 42 |
| 6 | 69/69 | 0 | 0 | 26 | 43 |
| 7 | 69/69 | 0 | 0 | 25 | 44 |
| 8 | 69/69 | 0 | 0 | 25 | 44 |

Every cycle map yielded exactly 69 unique, sequential, non-duplicated positions (1..69) with
zero ambiguous cells and zero unparsed/stray tokens — i.e. every word on each of the 8 grid
pages was consumed and accounted for by the parser.

**Cross-checks performed:**
- *Fresh-type set check*: for cycles 2-8, the set of assembly types found in "fresh" cells
  (blank `prev_position`) was compared against the feed-batch types declared in Table 6.1 for
  that cycle. **7 of 8 cycles matched exactly.** Cycle 5 showed one extra fresh-looking cell:
  position 40, type `E2` (E2 was cy3's feed type, not cy5's) with no `y` printed. This was
  hand-verified against the raw PDF word coordinates (not a parser artifact — the source
  literally prints no `y` for that cell) and is almost certainly the source's convention for
  "assembly did not move" (previous location = current location, so the redundant label is
  omitted) rather than a genuine fresh assembly. It is the only such case found in all 8×69 =
  552 cells. Left as-extracted (blank), not inferred/filled in, per the "no interpretation"
  instruction.
- *Batch-size checksum*: summing Table 6.1's per-type counts for each cycle's feed batch
  reproduces Table 8.2's independently-reported "No. of Feed FA" exactly for all 8 cycles
  (cy1=241, cy2=80, cy3-8=92) — see `write_tables.py` logic reproduced below. This validates
  Table 6.1 extraction against an independent summary table in the same document.
- *Cross-cycle chain continuity*: for cycles 2-8, each shuffled cell's `prev_position` was
  looked up in the prior cycle's map and its type compared. **546 of 552 (99%) matched.** 6
  mismatches (cy5: 1, cy7: 2, cy8: 3) show a position's type changing between cycles for what
  should be the same physical assembly (e.g. cy4 position 60 = `E1`, but cy5 position 20 cites
  `prev_position=60` with type `F3`). Each flagged mismatch was spot-checked against raw word
  coordinates and confirmed to be a faithful transcription of what the PDF prints — these are
  either genuine assembly-type relabeling in the 1999 source document (e.g. Gd-content
  subtype changes as poison burns out) or a source data inconsistency; they are **not**
  parser errors. Reported, not corrected.
- Quarter/octant raw type counts (unweighted, i.e. counting each of the 69 diagram positions
  once) are roughly full-core-count / 3.2-4.0 per type, consistent with a quarter-core
  symmetric map where most positions represent 4 physical assemblies and a minority on the
  symmetry axes/center represent fewer. Full symmetry-weighted reconciliation to the Table 6.1
  full-core counts was **not** attempted (it requires reconstructing each position's row/column
  and axis membership, which is more than the requested `position, prev_position, type` schema
  needs); the batch-size checksum above (which does not require per-position weights) was used
  instead as the primary cross-check.

### 2. Table 6.1 Fuel Assembly Data → `fuel_assembly_data.csv`

Source: Table 6.1 "Fuel Assembly Data for KNGR Multicycle", PDF page 15. Columns: cycle,
fuel assembly type, number of assemblies, fuel rod enrichment (w/o, `zone1/zone2` where
zoned), fuel rods per assembly (`type1/type2` where zoned), number of Gd shim rods, Gd2O3
weight percent. Straightforward columnar text, parsed by inspection (26 rows, cycles 1-8,
types A0/B0/B1/B2/C0/C1 (cy1) through J0/J2/J3 (cy8)). Verified: row sums per cycle equal
Table 8.2's "No. of Feed FA" exactly (see above).

### 3. Cycle physics summary → `cycle_summary.csv`

One row per cycle (1-8), columns sourced from three independent result tables so values could
be cross-checked against each other:

- `cycle_length_EFPD_table8.1`, `cycle_burnup_MWD_MTU` ← **Table 8.1** "Loading Pattern and
  ROCS Results for Each Cycle" (p.85) — decimal-precision EFPD and cycle-end core burnup.
- `cycle_length_EFPD_table8.2`, `cycle_length_EFPH`, `no_feed_FA`, `U_mass_MTU`,
  `core_avg_enrichment_wo`, `Fq_max_raw`, `Fxy_max_raw`, `Fr_max_raw`,
  `max_rod_burnup_MWD_MTU` ← **Table 8.2** "Core depletion calculation from cy1 to cy8"
  (p.86-87) — integer-rounded EFPD (cross-checks table 8.1's decimal value within <1 day for
  every cycle), plus the raw (unbiased) cycle-maximum Fq/Fxy/Fr and peak rod burnup.
- `CBC_BOC_ppm_HFP_NOXE`, `CBC_EOC_ppm_HFP_NOXE` ← **Table 6.43** "CBC and CDROM files from
  Cycle 1 to cycle 8" (p.63), HFP/no-Xe condition — this is the cleanest available BOC/max and
  EOC/min critical-boron figure per cycle (boron is monotonically decreasing over a cycle, so
  BOC = cycle max).
- `IBW_*`, `MTC_*` (BOC/EOC × EQ-Xe/NO-Xe) ← **Table 6.42** "IBW & MTC from cycle 1 to cycle 8"
  (p.62).

**Coverage**: cycle length (EFPD, MWD/MTU, EFPH) — found; boron BOC/EOC — found (Table 6.43,
HFP/NO-Xe only; the EQ-Xe boron series exists only embedded in the per-maneuver-step Tables
6.26-6.33, not as a clean per-cycle BOC/EOC summary, and was not separately extracted);
peaking factors Fr/Fxy/Fq — found (raw, cycle-maximum, from Table 8.2); IBW and MTC — found
as a bonus (Table 6.42, directly relevant to the design criterion "most positive MTC(HFP) <
0"). **Discharge burnup** — the document does not appear to contain a batch-average discharge
burnup table; the closest available figure is `max_rod_burnup_MWD_MTU` (Table 8.2's
"Max. Rod Burnup", the single highest-burnup rod reached during that cycle's depletion, not a
batch-average discharge value). This is flagged as a gap below rather than substituted
silently.

Per-maneuver-step (BOC→EOC, ~18-22 steps per cycle) reactivity/CBC tables (6.2-6.9, 6.26-6.33)
and per-maneuver-step best-estimated peaking-factor tables (6.34-6.41) exist and were read to
derive/verify the cycle-level maxima above, but the full step-by-step series was **not**
transcribed into CSV (out of the requested "cycle physics summary" scope; available on
request — all fully text-extractable, same table format as `pattern_cy0N.csv` inputs).

### 4. Methodology notes → `methodology_notes.md`

Short excerpts (not full transcription) with page citations covering: computer codes used
(§3.2, confirms ROCS 5.1 is the depletion code, run via CORD/ROCSEDIT/NCONEX/CENTAUR),
overall methodology (§5), the loading-pattern-search procedure and design criteria (§6.1),
the ROCS model reuse approach (§6.3), the radial albedo iteration decision — PVNGS albedo set
reused as-is, no independent iteration performed for KNGR (§6.4), the depletion-calculation
data products per cycle (§6.5), and the final evaluation/acceptance discussion including the
two flagged (not corrected) shortfalls: cy2 Fxy = 1.5697 > 1.55 target, and cycle lengths
slightly under the 468 EFPD target (§6.6).

## Known gaps (resisted text extraction)

These pages/figures are image-based (raster) in the source PDF; only surrounding
labels/legends extract as text, not the depicted data:

- **Figure 6.1** "Fuel Assembly Configurations" (PDF p.65) — pin-by-pin assembly layout
  diagrams (guide tube / normal / low-enriched / Gd pin legend extracts, the diagrams
  themselves do not).
- **Figures 6.10-6.17** "Axial Power Distribution for KNGR Cycle N" (PDF pp.74-81) — BOC/MOC/
  EOC axial power-shape line charts. Axis tick labels and the BOC/MOC/EOC legend extract as
  scattered numbers/text, but the plotted curve data does not. The underlying numeric data for
  these plots does exist in text form in **Tables 6.18-6.25** "Axial Power Distribution at HFP
  Equilibrium Condition" (pp.16-65 range) — those tables were located but not parsed into CSV
  (outside the requested scope).

No other pages in the 66-73 (loading maps) or 15 (Table 6.1) ranges showed extraction
resistance — both parsed cleanly and were independently checksummed as described above.

## Files in this directory

| File | Rows | Description |
|---|---|---|
| `pattern_cy01.csv` … `pattern_cy08.csv` | 69 each | Loading/shuffle map per cycle |
| `fuel_assembly_data.csv` | 26 | Table 6.1, fuel assembly types/enrichments/Gd per cycle |
| `cycle_summary.csv` | 8 | Per-cycle length, burnup, boron, peaking factors, IBW/MTC |
| `methodology_notes.md` | — | Short excerpts + page citations, §5-6 methodology |
| `README.md` | — | This file |

## Reproducing / auditing this extraction

All work was done with ad-hoc Python scripts (PyMuPDF) against the read-only source PDF;
no intermediate scripts were saved into this reference directory (they lived in a scratch
location). The key technique worth preserving for any future re-extraction: use
`page.get_text("words")` (not plain `get_text()`) for the loading-pattern-map pages, filter to
tokens matching `^\d+$` (position/rotation) or `^[A-J][0-3]$` (assembly type) within the y0
range ~170-648pt (excludes header/legend above and footer/caption below), and walk the
coordinate-ordered token stream with a monotonically-incrementing "expected position" counter,
using vertical offset (not token value) to distinguish the optional rotation digit from the
next cell's position label.
