# cond_v4 physics harvest — coverage & range report

_Generated 2026-07-18 from `data/store/fuel_types.parquet` — 120 rows x 34 columns (17 pre-existing + 17 cond_v4)._

HARVEST-side rebuild of the cond_v4 feature expansion. Sources: DeCART `FA_*.sum` (SUMMARY EDIT 1/2/3), `FA_*.HGC` (%TITL / %MACX / %ADFT / %DIST), and `dec_FA_*.inp` octant census. Ranges below feed the verify agent's normalization constants.

## Rows per library

| library | rows | physics-filled | note |
|---|--:|--:|---|
| 5.8_5.1 | 24 | 24 | 18 real lattices + 6 X-aliases (share base physics) |
| 260624 | 12 | 12 | all lattices |
| CPHA | 12 | 12 | all lattices |
| ga80 | 70 | 36 | 36 hydrated HGCs of 70 (34 stay manual-anchor `feature_poor`) |
| legacy_a | 2 | 0 | hard-coded A0/A1 anchors, no source files |
| **total** | **120** | **84** | |

## Fill counts per new column per library

| column | source | 5.8_5.1 | 260624 | CPHA | ga80 | legacy_a | ALL |
|---|---|--:|--:|--:|--:|--:|--:|
| `kinf0` | .sum EDIT2 K-CONV / HGC %TITL | 24 | 12 | 12 | 36 | 0 | **84** |
| `kinf10` | .sum EDIT2 K-CONV / HGC %TITL | 24 | 12 | 12 | 36 | 0 | **84** |
| `kinf20` | .sum EDIT2 K-CONV / HGC %TITL | 24 | 12 | 12 | 36 | 0 | **84** |
| `kinf30` | .sum EDIT2 K-CONV / HGC %TITL | 24 | 12 | 12 | 36 | 0 | **84** |
| `bu_k1` | root(k-inf=1) on ref curve | 24 | 12 | 12 | 36 | 0 | **84** |
| `ff_pin_max` | .sum EDIT3 FRP@BU=0 / HGC %DIST map1 max | 24 | 12 | 12 | 36 | 0 | **84** |
| `xs_d1` | HGC %MACX BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `xs_d2` | HGC %MACX BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `xs_a1` | HGC %MACX BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `xs_a2` | HGC %MACX BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `xs_nf1` | HGC %MACX BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `xs_nf2` | HGC %MACX BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `xs_s12` | HGC %MACX BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `adf_face_g1` | HGC %ADFT BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `adf_face_g2` | HGC %ADFT BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `adf_corner_g1` | HGC %ADFT BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `adf_corner_g2` | HGC %ADFT BOC | 24 | 12 | 12 | 36 | 0 | **84** |
| `boron_worth` | .sum/HGC BORON branch | 24 | 12 | 12 | 36 | 0 | **84** |
| `doppler_coef` | .sum/HGC TFUEL branch | 24 | 12 | 12 | 36 | 0 | **84** |
| `mtc_dmod` | .sum/HGC DMOD1-6 fit | 24 | 12 | 12 | 36 | 0 | **84** |
| `cr1_worth` | .sum/HGC CR1 REFERENCE | 24 | 12 | 12 | 36 | 0 | **84** |
| `zone_pin_count` | dec_FA_*.inp octant | 24 | 12 | 12 | 0 | 0 | **48** |

- **Curve + coefficients + xs + adf + ff (n=84):** every lattice row (24+12+12, incl. the 6 X-alias copies) + 36 ga80 HGC rows. The 5.8_5.1 B-series ships a state-point-suffixed `FA_B0x_0101.HGC`; the harvester resolves both `FA_<t>.HGC` and `FA_<t>_*.HGC`.
- **zone_pin_count (n=48):** lattice only (24+12+12); ga80 + legacy ship no `dec_FA_*.inp` -> NaN by contract.
- **legacy_a / 34 unfilled ga80:** no source file -> all cond_v4 columns NaN, `feature_poor` unchanged.

## Measured ranges (harvested rows only)

| column | unit | min | median | max | n |
|---|---|--:|--:|--:|--:|
| `kinf0` | - | 1.09124 | 1.15753 | 1.26046 | 84 |
| `kinf10` | - | 1.07157 | 1.11461 | 1.19715 | 84 |
| `kinf20` | - | 1.10385 | 1.12161 | 1.16415 | 84 |
| `kinf30` | - | 1.05779 | 1.08301 | 1.09367 | 84 |
| `bu_k1` | GWd/tU | 38.54148 | 42.6163 | 44.37082 | 84 |
| `ff_pin_max` | - | 1.101 | 1.1435 | 1.2016 | 84 |
| `xs_d1` | cm | 1.42056 | 1.43113 | 1.43726 | 84 |
| `xs_d2` | cm | 0.49251 | 0.49537 | 0.49794 | 84 |
| `xs_a1` | 1/cm | 0.01 | 0.0103 | 0.01052 | 84 |
| `xs_a2` | 1/cm | 0.10313 | 0.10906 | 0.11375 | 84 |
| `xs_nf1` | 1/cm | 0.00754 | 0.00793 | 0.00809 | 84 |
| `xs_nf2` | 1/cm | 0.14177 | 0.1525 | 0.1613 | 84 |
| `xs_s12` | 1/cm | 0.01644 | 0.01669 | 0.01697 | 84 |
| `adf_face_g1` | - | 0.99291 | 1.00162 | 1.02025 | 84 |
| `adf_face_g2` | - | 1.02338 | 1.07136 | 1.14588 | 84 |
| `adf_corner_g1` | - | 0.95204 | 0.97111 | 1.0096 | 84 |
| `adf_corner_g2` | - | 1.1692 | 1.23645 | 1.30483 | 84 |
| `boron_worth` | pcm/ppm | -5.91375 | -5.48938 | -5.3403 | 84 |
| `doppler_coef` | pcm/K | -2.12806 | -1.97735 | -1.77602 | 84 |
| `mtc_dmod` | pcm/0.01gcc | 72.19113 | 104.71691 | 122.92253 | 84 |
| `cr1_worth` | pcm | 11455.85301 | 12414.31658 | 12872.21789 | 84 |
| `zone_pin_count` | pins | 52.0 | 76.0 | 100.0 | 48 |

### Per-library kinf0 spread

| library | kinf0 min | kinf0 max |
|---|--:|--:|
| 5.8_5.1 | 1.1503 | 1.2605 |
| 260624 | 1.1104 | 1.177 |
| CPHA | 1.1503 | 1.2202 |
| ga80 | 1.0912 | 1.1794 |

## Sign / convention conformance (column contract)

| coefficient | contract sign | observed [min, max] | status |
|---|---|---|---|
| `boron_worth` | negative | [-5.91375, -5.3403] | OK |
| `doppler_coef` | negative | [-2.12806, -1.77602] | OK |
| `cr1_worth` | positive | [11455.85301, 12872.21789] | OK |
| `mtc_dmod` | as-computed (+, denser moderator -> +rho) | [72.19113, 122.92253] | OK |

## Cross-checks

- **.sum <-> HGC parity** on all 5.8_5.1 lattices carrying both products: `kinf0` agree < 1e-4, `ff_pin_max` (FRP vs %DIST map1 max) agree < 0.01.
- **octant census <-> dir n_gd** on all 42 lattice `dec_FA_*.inp`: 0 mismatches (multiplicity x4 on diagonal / x8 off-diagonal, sum = 256; zoning `UO2_2` pins counted separately from Gd `UO2G`).
- **glued-negative parse:** CR1 / high-burnup blocks print a negative buckling with no separating space (`...E-01-6.29045E-04`); a scientific-float regex recovers both fields. This fixed a silent drop of ALL 17 CR1-REFERENCE blocks + the BU>38 reference points in ga80 HGCs (before the fix, ga80 `cr1_worth` was NaN for 22/36 and `bu_k1` NaN for 36/36).
- **byte-identical guard:** the 17 pre-cond_v4 columns are unchanged; pre-existing `source_flags` stay a strict prefix of the rebuilt lists.

## Notes for normalization

- `kinf20` min (1.104) is the Gd-burnout hump peak of a heavily-loaded ga80 "A" design (`kinf20 > kinf0`); k-inf curves are only monotone-declining *after* the hump, so a per-point normalization must not assume `kinf0` is the curve maximum.
- `zone_pin_count` is bimodal by axial zone: 52 pins (z1) vs 100 pins (z2).
- ga80 reference sweeps run the full 62-point curve to 80 GWd/tU, so every filled row has a finite `bu_k1` in [38.5, 44.4] GWd/tU.
