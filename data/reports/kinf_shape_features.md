# k-conv (k-inf vs burnup) curve-shape features — coverage, distributions & v5 channel plan

_Generated 2026-07-20 from `data/store/fuel_types.parquet` — 153 rows x 55 columns (46 pre-existing + 9 k-conv shape). Harvest code: `lpopt/data/fuel_types.py::kconv_curve_shape`, wired into `_curve_and_coeffs` so the `.sum` (EDIT2 K-CONV) and HGC (%TITL k-inf per state) paths fill it identically._

The k-conv shape features summarise the **burnable-absorber holddown -> release SIGNATURE** read off the reference k-inf(BU) depletion curve, in reactivity space `rho = (k-1)/k * 1e5` [pcm]. They describe the *shape* of the curve, not the poison design that produced it, so they are **poison-agnostic**: the same dip / hump / swing / slope descriptors apply to a Gd lattice today and to an IFBA / Er / Dy assembly tomorrow. This is the property that makes the curve shape the universal absorber feature for the v5 retrain (see "v5 channel plan" below).

## Columns added (all additive nullable float, appended after `v_mod_over_v_fuel`)

| column | unit | meaning | monotone-curve value |
|---|---|---|---|
| `kinf_dip` | - | k-inf at the suppression trough (local min preceding the burnout hump) | NaN |
| `bu_dip_gwd` | GWd/tU | burnup at the trough | NaN |
| `kinf_peak` | - | k-inf at the post-dip burnout hump maximum (absorber fully burned out) | `kinf0` (BU=0) |
| `bu_peak_gwd` | GWd/tU | burnup at the hump | 0.0 |
| `reactivity_swing_pcm` | pcm | `rho_peak - rho_dip` — holddown-release magnitude | NaN |
| `rho_boc_minus_peak_pcm` | pcm | `rho(0) - rho_peak` — BOC reactivity vs hump (xenon-free fresh boost sign) | 0.0 |
| `depletion_slope_pcm_per_gwd` | pcm/GWd | least-squares `d(rho)/dBU` over `bu_peak..min(60, last)` (BU=0 fresh point excluded) | valid (peak=BU0) |
| `kinf_eol50` | - | k-inf interpolated at 50 GWd/tU | valid |
| `kconv_is_monotone` | flag | 1.0 = no prominent hump (weak/absent absorber), 0.0 = hump present | 1.0 |

Hump detection: the first prominent interior local maximum (peak above its preceding trough by >= `1e-4` in k ~ 8 pcm), with the running minimum up to it as the dip. The BU=0 xenon-free spike is a decreasing leg and never registers as a dip or peak. A curve with no such hump is `kconv_is_monotone=1` and leaves `kinf_dip / bu_dip_gwd / reactivity_swing_pcm` NaN by omission.

## Coverage per library

Same `(library_id, type_id)` fill as `kinf0` — every harvested reference curve gets a `kinf_peak` / slope / eol50; the dip/swing subset only fills where a prominent hump exists.

| library | rows | curve (kinf_peak) | hump (dip/swing) | monotone |
|---|--:|--:|--:|--:|
| 5.8_5.1 | 24 | 24 | 14 | 10 |
| 260624 | 12 | 12 | 12 | 0 |
| CPHA | 12 | 12 | 10 | 2 |
| ga80 | 70 | 36 | 36 | 0 |
| paramA | 33 | 33 | 20 | 13 |
| legacy_a | 2 | 0 | 0 | 0 |
| **total** | **153** | **117** | **92** | **25** |

- **curve (n=117):** every row that carries a reference k-inf curve (all lattices incl. the 6 X-alias copies, 36 hydrated ga80 HGCs, 33 paramA HGC-only). Identical set to `kinf0`.
- **hump (n=92):** rows with a prominent burnout hump. The 25 monotone rows are the weak-absorber designs — dominated by the 6wt%/12pin (and some 6wt%/16pin) Gd families whose k-inf declines monotonically after the xenon transient.
- **legacy_a (n=0):** no source curve -> all 9 columns NaN, same contract as `kinf0`.
- paramA rows require the augment to run with `paramA_root = data/design/package` resolved (33 HGC-only designs); otherwise those keys keep NaN.

## Fill counts per column per library

| column | 5.8_5.1 | 260624 | CPHA | ga80 | paramA | legacy_a | ALL |
|---|--:|--:|--:|--:|--:|--:|--:|
| `kinf_dip` | 14 | 12 | 10 | 36 | 20 | 0 | **92** |
| `bu_dip_gwd` | 14 | 12 | 10 | 36 | 20 | 0 | **92** |
| `kinf_peak` | 24 | 12 | 12 | 36 | 33 | 0 | **117** |
| `bu_peak_gwd` | 24 | 12 | 12 | 36 | 33 | 0 | **117** |
| `reactivity_swing_pcm` | 14 | 12 | 10 | 36 | 20 | 0 | **92** |
| `rho_boc_minus_peak_pcm` | 24 | 12 | 12 | 36 | 33 | 0 | **117** |
| `depletion_slope_pcm_per_gwd` | 24 | 12 | 12 | 36 | 33 | 0 | **117** |
| `kinf_eol50` | 24 | 12 | 12 | 36 | 33 | 0 | **117** |
| `kconv_is_monotone` | 24 | 12 | 12 | 36 | 33 | 0 | **117** |

## Measured distributions (harvested rows only)

| column | unit | min | median | max | n |
|---|---|--:|--:|--:|--:|
| `kinf_dip` | - | 1.0592 | 1.1142 | 1.1996 | 92 |
| `bu_dip_gwd` | GWd/tU | 0.5000 | 7.0000 | 18.0000 | 92 |
| `kinf_peak` | - | 1.1059 | 1.1461 | 1.2933 | 117 |
| `bu_peak_gwd` | GWd/tU | 0.0000 | 19.0000 | 25.0000 | 117 |
| `reactivity_swing_pcm` | pcm | 9.0330 | 1046.8073 | 5197.7235 | 92 |
| `rho_boc_minus_peak_pcm` | pcm | -2749.0802 | 46.6063 | 4125.0851 | 117 |
| `depletion_slope_pcm_per_gwd` | pcm/GWd | -650.7013 | -599.8500 | -340.2156 | 117 |
| `kinf_eol50` | - | 0.9271 | 0.9566 | 1.0069 | 117 |
| `kconv_is_monotone` | flag | 0.0 | 0.0 | 1.0 | 117 |

Physics sanity confirmed on the reference families (`tests/test_fuel_kconv_shape.py`):
- **strong-Gd 10wt%/20pin** (e.g. 5.8_5.1 `B05`, 260624 `B5`): dip 1.1144 < peak 1.1238, `bu_dip=12` (in [5,20]), `bu_peak=22` (in [15,30]), swing +755 pcm, slope -600 pcm/GWd.
- **weak-Gd 6wt%/12pin** (e.g. 5.8_5.1 `X0`): `kconv_is_monotone=1`, dip/swing NaN, peak degenerates to `kinf0`, slope still valid (-470 pcm/GWd). Handled gracefully.
- **.sum vs HGC parity** on the 10/20 hump type: `bu_dip`/`bu_peak` identical, `kinf_dip`/`kinf_peak` agree < 1e-4, swing agree < 0.4 pcm, slope agree < 0.003 pcm/GWd.

## v5 channel plan (poison-agnostic feature set)

> **Directive (2026-07-20):** the training FEATURE set must become poison-agnostic — future assemblies use other burnable absorbers (IFBA/Er/Dy). Poison-SPECIFIC design channels must NOT be model inputs. The k-inf curve shape becomes the universal poison signature.

This note **registers the candidate channels** for the completion-bundle v5 retrain. The v5 schema itself (rank loss + physics prior + quantile heads) is NOT created here — it lands with that bundle; `featurize.py` is untouched. The candidate inventory:

**REMOVE from the channel inventory (poison-specific design axes):**
- cell channels `origin_n_gd`, `origin_gd_wt`, `origin_gd_u_enr`
- globals `g_fresh_mean_n_gd`, `g_fresh_mean_gd_wt`

**ADD (poison-agnostic curve-shape channels, traced to the fresh chain origin exactly like the existing `origin_*` block; None/NaN -> 0.0 with a presence gate):**

| proposed channel | source column | normalization `(x - ref)/scale` | note |
|---|---|---|---|
| `origin_reactivity_swing` | `reactivity_swing_pcm` | `swing / 2500` (NaN->0) | holddown-release magnitude; **physics-prior input** |
| `origin_depletion_slope` | `depletion_slope_pcm_per_gwd` | `(slope + 600) / 130` | burnout decay rate; **physics-prior input** |
| `origin_bu_peak` | `bu_peak_gwd` | `(bu_peak - 19) / 12` | burnout timing |
| `origin_bu_dip` | `bu_dip_gwd` | `(bu_dip - 7) / 7` (NaN->0) | suppression-trough timing |
| `origin_rho_boc_minus_peak` | `rho_boc_minus_peak_pcm` | `rho_bmp / 2800` | initial suppression depth (signed) |
| `origin_kinf_eol50` | `kinf_eol50` | `(k - 0.957) / 0.05` | discharge reactivity |
| `origin_kconv_monotone` | `kconv_is_monotone` | flag {0,1} | has-hump gate (poison strength) |
| `origin_kconv_present` | (derived) | 1.0 when `kinf_peak` finite | presence gate for the block |

Normalization constants are `ref = median`, `scale ~= robust half-span (p95-p05)/2` rounded from the measured population above; refine at v5 build once the completion corpus is final. `kinf_dip`/`kinf_peak` are intentionally NOT separate channels — they are redundant with `reactivity_swing_pcm` (dip->peak in rho-space) plus `kinf_eol50`; they remain in `fuel_types` for QC.

**ADD (globals, replacing the removed poison-specific means — feed-weighted fresh-slot means, mirroring the old `g_fresh_mean_n_gd`):**
- `g_fresh_mean_reactivity_swing` — whole-core fresh-feed absorber-release summary
- `g_fresh_mean_depletion_slope` — whole-core fresh-feed burnout-decay summary

**KEEP:** all other v4 result-value channels (`origin_kinf0/10/20/30`, `origin_bu_k1`, `origin_boron_worth`, `origin_doppler`, `origin_mtc_dmod`, `origin_cr1_worth`, `origin_ff_pin_max`, `origin_xs_*`, `origin_adf_*`, `origin_zone_pins`, `origin_enr_*`, `origin_u_mass`, `origin_lattice_present`, etc.) — these are result/behavioural signatures, not poison design axes.

**`fuel_types` KEEPS the Gd columns** (`n_gd`, `gd_wt`, `gd_u_enr`) as bookkeeping / QC metadata — no columns are dropped from the store. Only the *model channel selection* drops them.

### Physics-prior linkage
The adopted physics-prior work consumes `reactivity_swing_pcm` and `depletion_slope_pcm_per_gwd` **directly**: swing bounds the reactivity a fresh absorber-bearing assembly can release into the cycle, and the depletion slope sets the burnout-region reactivity decay that drives cycle length. Both are poison-agnostic, so the prior transfers to any absorber chemistry without re-derivation.

### Ablation A/B requirement (v5 retrain)
Run a remove-Gd vs keep-Gd ablation at the v5 retrain: (A) drop `origin_n_gd/gd_wt/gd_u_enr` + the two Gd globals and add the shape channels; (B) keep the Gd channels alongside the shape channels. **Gate:** (A) must show **no in-Gd-corpus accuracy loss** vs (B) — i.e. the curve-shape features fully carry the poison information the Gd design axes carried, confirming the shape set is a lossless, generalizing replacement.

### OOD-guard implication
The per-channel training-population feature envelope (`model_api` sidecar, review sec. 4b) already covers the shape columns' ranges (tabulated above). Because the poison axis is now represented by **shape-feature envelopes** rather than `n_gd`/`gd_wt` bounds, the OOD guard flags out-of-distribution absorber behaviour for **any** absorber type (an IFBA/Er/Dy assembly with an unfamiliar swing/slope/hump-timing trips the envelope), instead of only flagging Gd loadings outside the trained pin-count/wt% grid. The poison OOD axis becomes chemistry-independent.
