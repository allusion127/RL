# Pin pitch / pin radius optimization — viability & readiness review

_Evidence-based scoping for the planned future core optimization that varies **pin pitch** and **pin radius** while holding the assembly envelope fixed. Written for the implement phase: section 4 is the executable spec._

Sources read: `lpopt/model/featurize.py` (`_V4_SCALES`, channel inventory), `lpopt/data/fuel_types.py` (harvest + schema), `lpopt/design/lattice.py` (dec deck edit chain), `lpopt/design/coredeck.py` (MASTER deck), `lpopt/model/model_api.py` (`unresolved_fresh_types`, channel-parity gate), `lpopt/report/report.py` (campaign report), `data/curriculum/transfer_curve.json` (blind-transfer record), `data/reports/fuel_types_v4_harvest.md` (population ranges), a real lattice deck `data/design/curriculum_work/5.75-6_f109/S3/dec_FA_S3.inp` (GEOM card), and `data/store/fuel_types.parquet` (131×39, u_mass n=59).

---

## 0. The load-bearing constraint, verified

The DeCART `GEOM` card carries **both** pitches on one line (`dec_FA_S3.inp:66-68`):

```
GEOM
 npins 16
 pitch 1.285 20.7772          <- token1 = PIN pitch, token2 = ASSEMBLY pitch
 cellgeo 1 0.4096 0.4178 0.4750 / 5 1 1     ! r_pellet, r_clad_in, r_clad_out (normal pin)
 cellgeo 2 0.4096 0.4178 0.4750 / 10 1 1    ! IGD pin (same radii)
 cellgeo 3..6 0.57150 ... 1.24450           ! guide tubes (FIXED)
```

The assembly-pitch token `20.7772` is **byte-identical** to `CoreParams.wide = 20.7772` in the MASTER deck (`coredeck.py:79`, emitted into `%GEN_GEO`). This is the single linkage that keeps the MASTER core model valid.

**Constraint (VERIFIED and to be stated explicitly in the plan):** the assembly envelope is fixed **iff** the *second* `pitch` token stays `20.7772` **iff** MASTER `%GEN_GEO wide` is unchanged. The optimization may edit only the **first** `pitch` token (pin pitch) and the `cellgeo 1`/`cellgeo 2` radii. Guide-tube `cellgeo 3-6`, `npins 16`, and token2 stay frozen. Under that rule the MASTER nodal core model, its mesh, and restart files remain valid (section 3). If a later plan ever varies token2, the MASTER core geometry is invalidated wholesale — that is the boundary of this review.

---

## 1. Physics scoping

### 1.1 Geometric admissibility (16×16, guide tubes fixed, assembly pitch fixed)

Pin array footprint = `16 × 1.285 = 20.560 cm`; assembly pitch `20.7772 cm`; **leftover water span = 0.2172 cm total (0.1086 cm per side)**. That thin peripheral water gap is the binding constraint on pitch increase:

| axis | direction | geometric limit | admissible band |
|---|---|---|---|
| pin pitch | **up** | `20.7772/16 = 1.29858` → array fills the node, zero inter-assembly water (unphysical) | **+1.06 % absolute max; ~+0.5 % to keep half the water gap** |
| pin pitch | down | pitch > clad OD diameter `0.950 cm` (rods touch across boundary) | to ~−26 % (moderation collapses long before) |
| pin radius (pellet+clad co-scaled) | up | clad OD < pitch (rods touch within lattice) | to ~+35 % geometrically |
| pin radius | down | none practical (thinner rods, more water) | wide |

**Hard finding:** a symmetric ±3 % pitch grid is **not admissible**. Pin pitch has a ~+1 % ceiling and a wide floor — the pitch design space is strongly asymmetric. Any grid the implement phase builds must respect `pin_pitch ≤ 1.298 cm` (better ≤ ~1.292). Radius has room both ways geometrically, but is capped far earlier by normalization/OOD (section 1.3).

### 1.2 Moderation ratio and the harvested features

Single-cell moderation ratio (Vm/Vf, guide tubes ignored) nominal = **1.788**. Elasticities: `dln(Vm/Vf)/dln(pitch) ≈ +3.6`, `dln(Vm/Vf)/dln(radius_coscaled) ≈ −3.4`.

| perturbation (nominal ±3 % for magnitude, note pitch +3 % is inadmissible) | Vm/Vf | Δ | direction of spectrum |
|---|---|---|---|
| pitch +3 % | 1.979 | +10.7 % | softer / more thermal |
| pitch +1 % (admissible edge) | 1.851 | +3.5 % | slightly softer |
| pitch −3 % | 1.603 | −10.4 % | harder |
| radius +3 % (co-scaled) | 1.608 | −10.1 % | harder (**+u_mass 6.1 %**) |
| radius −3 % (co-scaled) | 1.985 | +11.0 % | softer (**−u_mass 5.9 %**) |

PWR lattices sit **under-moderated** at operating conditions (this is why MTC is negative and adding moderator raises reactivity). So more moderation (pitch up / radius down) → higher thermal utilization, less resonance capture. Directional + rough-magnitude response of each harvested feature to **more moderation** (softer spectrum), for ~+3.5 % Vm/Vf (i.e. the admissible +1 % pitch, or −1 % radius):

| harvested feature | dir. (more moderation) | rough Δ at +3.5 % Vm/Vf | why |
|---|---|---|---|
| `kinf0..kinf30`, `bu_k1` | ↑ | +0.5–1.5 % k (≈ +500–1500 pcm) | under-moderated → +moderation raises reactivity |
| `xs_s12` (1→2 downscatter) | **↑ strongly** | **+2–4 %** | downscatter ∝ H (moderator) volume — near-linear |
| `xs_a2`, `xs_nf2` (thermal macro) | ↓ | −2–4 % | fuel volume fraction dilutes |
| `mtc_dmod` | ↓ | tens of pcm | moving toward optimal moderation lowers dρ/dρ_mod |
| `boron_worth` (pcm/ppm, <0) | ↑ magnitude | +5–10 % | softer thermal flux → boron 1/v more effective |
| `doppler_coef` (<0) | ↓ magnitude | small | less U-238 resonance weight |
| `adf_corner_g2` | shifts notably | ~+3–8 % | water-gap change drives thermal edge peaking |
| `ff_pin_max` | mild reshape | small | intra-assembly flux shape |
| `u_mass_g` | **0 for pitch, ∝ r² for radius** | radius only | U inventory is per pellet, pitch-independent |

Radius and pitch are **distinguishable axes**: pitch moves spectrum with `u_mass` frozen; radius moves spectrum **and** `u_mass` together.

### 1.3 Which normalization constant breaks first

`_V4_SCALES` / the k-inf & u_mass constants (`featurize.py:114-139`) were fit to the Gd population (scale ≈ half-range). Confirmed population half-ranges vs scale:

| channel | ref | scale | scale as % of ref | fragility |
|---|---|---|---|---|
| `u_mass` | 138.8 | **0.7 g** | 0.5 % | **most fragile (radius canary)** |
| `xs_s12` | 0.0167 | **0.00025** | 1.5 % | **most fragile spectral (pitch canary)** |
| `xs_a2` | 0.109 | 0.005 | 4.6 % | moderate |
| `adf_corner_g2` | 1.236 | 0.07 | 5.7 % | moderate |
| `boron_worth` | −5.5 | 0.3 | 5.5 % | moderate |
| `xs_nf2` | 0.152 | 0.01 | 6.6 % | moderate |
| `doppler` | −2.0 | 0.2 | 10 % | robust |
| `mtc_dmod` | 105 | 25 | 24 % | robust |
| `kinf*` | 1.0 | 0.25 | 25 % | robust |

**Break-order under a moderation shift:**

- **Radius axis → `u_mass` breaks first.** `u_mass ∝ r_pellet²`, and the population half-range is only 0.7 g on a 138.8 g mean (store-confirmed: n=59, [138.07, 139.45]). So:
  - +0.5 % radius → u_mass 140.2 g → **z = +2.0**
  - +1.0 % radius → 141.6 g → **z = +4.0**
  - +3.0 % radius → 147.3 g → **z = +12.1**
  `u_mass` is a near-exact radius proxy, so a radius change is **loud** — it saturates the OOD guard almost immediately. Next to go: `xs_s12`, then `xs_a2`/`xs_nf2`.
- **Pitch axis → `xs_s12` breaks first.** Downscatter scales ~linearly with moderator volume; scale is only 1.5 % of ref, so even +2–3 % pitch (≈ +6 % water) pushes `xs_s12` past the population edge (~2–4 z). `adf_corner_g2` follows (water-gap driven). Critically, **`u_mass` does NOT move for pure pitch** → the pitch axis is *silent on the radius canary* and must be caught by the spectral channels.
- `kinf`, `mtc_dmod`, `doppler` are robust (wide scales) and will remain in-range across the entire admissible pitch band — they cannot be relied on as OOD tripwires.

---

## 2. Model-mechanism analysis

The v4 encoder feeds the model a **homogenized-lattice description** per fuel type (`featurize.py` `_V4_EXTRA`): kinf curve, branch coefficients (boron/doppler/mtc/cr1), 2-group macro XS (`xs_a2/nf2/s12`), `adf_corner_g2`, `ff_pin_max`, `u_mass`, `zone_pins`, plus enrichments/Gd. Because DeCART re-runs per variant, **these feature values genuinely reflect the new moderation** — the encoder will faithfully encode a pitch/radius change through the shifted feature block. So the question is not whether the input sees the change, but whether the learned feature→metric operator propagates it correctly.

### 2.1 Exposure by target

- **Spectrum / reactivity-mediated (cbc_max, cyclen, ao_abs):** carried by the branch/XS/kinf features the model provably leans on (cyclen blind-transfer Spearman 0.6–0.9 in `transfer_curve.json` is evidence the kinf/bu_k1 features carry lifetime). A moderation-driven feature shift moves these in the **correct direction** with roughly correct magnitude. These are the *recoverable* targets.
- **Geometry / form-function-driven (f_r, f_q, max_pin_burnup):** F_r is intra-assembly pin-power peaking, physically set by the full pin form function (HFF). The model sees only **one scalar** `ff_pin_max` plus the `pin_bu_*` curve summary — not the reshaped pin-power distribution. A pitch/radius change reshapes the *whole* form function (tighter lattice / changed water-gap edge peaking), but the model can only shift F_r by however `ff_pin_max` moved. **These are the least trustworthy under geometry variation**, even though the DeCART+MASTER-HFF *truth* captures them fully. `max_pin_burnup` is doubly exposed (served via the physics estimator off cyclen, `model_api.py:_pinbu_column`, itself keyed to fixed-geometry peaking curves).

### 2.2 What the model CAN vs CANNOT infer

**CAN (through result-features):** the sign and rough magnitude of the reactivity/lifetime response. This is the explicit design intent of physics-featurization — a new lattice "speaks" to the model in the same 2-group/ADF/kinf language MASTER uses, so a moderation move that DeCART renders into shifted features is at least directionally propagated to cbc/cyclen/AO.

**CANNOT structurally disentangle:** the moderation ratio as an independent axis. Two hard reasons:
1. **No moderation input channel.** There is no `pitch`, `radius`, or `Vm/Vf` feature today (`fuel_types` has no geometry columns — store-confirmed 0 geometry cols in 131×39). Moderation is visible only as a *shadow* on the homogenized features.
2. **Training features are collinear on a ~3-4 DOF manifold** (enrichment × Gd × zoning × residence-age). Along that manifold `kinf`, `xs_nf2`, `xs_a2`, `boron_worth` co-move with enrichment, and `xs_s12`/`mtc` barely move. A moderation change moves the feature vector in a **new direction** (kinf ↑ *with* xs_s12 ↑↑ *and* mtc ↓ *and* adf shifted — a covariance signature never seen). The model's learned combination of collinear proxies is under-determined off-manifold: features that were perfectly correlated in training can now move independently, and the network may double-count or cancel them. The most exposed reactivity target is **core MTC / AO**, whose mapping from the per-node `mtc_dmod` feature was only ever calibrated on the fixed-geometry manifold.

### 2.3 Failure mode (the crux)

Two regimes:

- **LOUD OOD (desirable):** radius changes (`u_mass` z blows out) and larger pitch changes (`xs_s12`/`adf` z blow out). The per-channel z-guard (section 4b) fires; the campaign quarantines. We *want* geometry moves to be loud, and for radius they intrinsically are.
- **SILENT in-range-but-wrong (dangerous):** small pitch changes (≲ ~2 %) whose feature shifts still land inside every channel's population envelope. No z-guard fires, yet the metric mapping is off-manifold. **Ensemble epistemic variance will NOT catch this** — all members trained on the same manifold agree with each other while being *jointly* wrong (disagreement measures interpolation uncertainty, not extrapolation bias). This is exactly why the model's own uncertainty cannot self-certify a moderation move, and why section 4c (DeCART blind-probe against MASTER truth) is mandatory rather than optional.

---

## 3. MASTER-side constraints

### 3.1 What stays valid (no regeneration)

With the section-0 constraint held, pin-level geometry lives **entirely inside the DeCART lattice**; the MASTER nodal model is invariant:

- `%GEN_GEO wide = 20.7772`, `height`, `zmesh`, `nz` — node width = assembly pitch, unchanged (`coredeck.py:135-157`).
- `%GEN_DIM` = `nx, ny, nz, nbatch, ncomp, ndim, ngeo, nsym, ndivxy, ndivz, ng` — **none encode pin geometry**; `nbatch/ncomp` change only with the number of fuel types (`_dims`, `coredeck.py:104-106`), exactly as an enrichment-only campaign already does. **No `%GEN_DIM` change from pin-level geometry.**
- `%GEN_PIN` = `icornf, iweigh, npin=16, nfrod=236` — fixed while 16×16 and the guide-tube layout are fixed (`coredeck.py:122-132`).
- `%GEN_SYM`, `%GEN_CDN`, `%LPD_BCH/SHF` maps — geometry-independent.
- **Restart compatibility:** `MAS_RST.*` carries node-wise burnup/number-densities on the fixed nodal mesh; since the mesh is unchanged, restart files remain valid across cycles (the `build_reload_deck` irrst=1 path, `coredeck.py:273`).
- **HFF reconstruction** with new per-type form functions is valid: `MAS_HFF` is a per-type product referenced as `FA_<alias>` (`coredeck.py:_lpd_static`), regenerated with the variant lattice like any new type.

### 3.2 What must be regenerated per variant

`MAS_XSL` (homogenized 2-group XS), `MAS_HFF` (pin form functions), and the `HGC`/`.sum`/`.out` products — the **same per-type products an enrichment change already regenerates**. The existing design chain covers this end-to-end:
- `lattice.py`: DeCART run + product harvest (`run_batch`, cap 4 parallel — matches the DeCART cap in scope).
- `fuel_types.py`: `harvest_lattice_cond_v4` re-derives the feature block from the new products.
- `coredeck.py`: references each type as `FA_<alias>` in `%LPD_C&X`/`%LPD_HFF`.

### 3.3 Hard blockers

**None for pin-level geometry**, provided section-0 holds. The only blocker is the boundary itself: changing the assembly pitch (token2) would change `%GEN_GEO wide`, invalidate the entire MASTER core geometry, and break restart compatibility. Keep it frozen and there is no `%GEN_DIM` change, no mesh change, no restart break.

One implementation gap to close: `lattice.py:edit_dec_text` today edits **MATERIAL only** (UO2/UO2_2 92235, UO2G 6408) and by docstring "all geometry stays byte-identical to the template" (`lattice.py:114-120`). To vary pin geometry it must additionally edit the **first** `pitch` token and the `cellgeo 1`/`cellgeo 2` leading radii — an additive edit, with a hard guard asserting token2 and `cellgeo 3-6` are untouched.

---

## 4. Verification procedure — implementation spec

### 4a. `fuel_types` geometry columns to harvest (additive-extend)

The store has **no** geometry columns today (confirmed: 0 of 39). Add these nullable-float columns (parsed from the dec inp `GEOM` block; **NaN for HGC-only types** — ga80/legacy ship no dec inp, so they stay NaN by the same contract as `zone_pin_count`):

| column | source (dec `GEOM`) | nominal |
|---|---|---|
| `pin_pitch` | `pitch` token1 | 1.285 |
| `asm_pitch` | `pitch` token2 (assertion anchor) | 20.7772 |
| `r_pellet` | `cellgeo 1` radius 1 | 0.4096 |
| `r_clad_in` | `cellgeo 1` radius 2 | 0.4178 |
| `r_clad_out` | `cellgeo 1` radius 3 | 0.4750 |
| `p_over_d` | `pin_pitch / (2·r_clad_out)` | 1.353 |
| `v_mod_over_v_fuel` | `(pitch² − π·r_clad_out²)/(π·r_pellet²)` | 1.788 |

Implementation: extend `parse_dir_geometry`/`rows_from_fa_dir` (`fuel_types.py:356, 1177`) with a `parse_dec_geom(path)` reading the `pitch` and `cellgeo 1` lines; add the columns to `SCHEMA_COLUMNS`, `FuelVec`, `_FLOAT_COLUMNS`, `_vec_from_row`. Use the additive `augment_fuel_table_*` pattern (`fuel_types.py:1497`) so existing rows are preserved byte-for-byte and legacy/ga80 keep NaN. These columns are **also** new featurizer channels + their own `_V4_SCALES` entries (a v4→v5 schema bump; see 4c retrain note).

### 4b. Serve-time OOD guard (per-channel population z-range)

Mirror `unresolved_fresh_types` (`model_api.py:410`) exactly — a method that returns a list of offending types and a warning surface, not a hard fail:

1. **Persist, at train time, a per-channel population envelope** `{channel: (z_min, z_max)}` computed over the training fuel population (each `z = (value − ref)/scale` using the same `_V4_SCALES`/u_mass/kinf constants, plus the new geometry channels). Store it in the checkpoint `backend.json` next to `channels`/`globals` (`model_api.py:save`).
2. **Add `PosValCnnBackend.feature_ood_types(pattern)`**: for each fresh type, compute every channel's z from its `FuelVec`; flag the type if any channel z falls outside `[z_min − m, z_max + m]` (`m ≈ 0.5`). Return `{type_id: [(channel, z), ...]}`. This catches both regimes: radius (u_mass z ≫ envelope) and pitch (xs_s12 z ≫ envelope) — including the pitch case where `u_mass` is nominal but the spectral channel is not.
3. **Campaign report integration:** surface it in `report.build_report` (`report.py:188`) as a warning block alongside the existing verified-LP section, identical treatment to how `unresolved_fresh_types` is meant to warn. Any pattern touching a flagged type is annotated "geometry/spectrum OOD — prediction unvalidated."

Rationale: the z-range guard, not ensemble variance, is the front line — section 2.3 shows epistemic variance is blind to off-manifold bias. The guard is loud by construction for radius and for pitch ≳ 2 %; the residual silent band (pitch ≲ 2 %) is covered by 4c.

### 4c. Pre-campaign GEOMETRY VALIDATION PROTOCOL (blind-probe transfer measurement)

Gate the entire geometry axis on a DeCART→MASTER transfer test **before** any optimizer consumes geometry-varied types. Stage in the registered scratch dir `C:/Users/USER/AppData/Local/Temp/eqlp_geomchk`.

1. **Grid.** N variant lattices over the **admissible** band (section 1.1): `pin_pitch ∈ {−3 %, −1 %, 0, +0.5 %}` (respect the +1 % ceiling; do NOT put +3 % on the grid) × `r_pellet/clad co-scaled ∈ {−3 %, −1 %, 0, +1 %}`, crossed with 2-3 representative enrichment/Gd anchors. Start N ≈ 12-16.
2. **DeCART.** Run via `lattice.run_batch` (cap 4 parallel, existing chain). Assert token2 = 20.7772 and `cellgeo 3-6` unchanged on every generated deck.
3. **Harvest.** `harvest_lattice_cond_v4` + the new 4a geometry columns.
4. **MASTER truth.** Build variant cores (`coredeck.build_reload_deck`) and run MASTER to get ground-truth (cbc, cyclen, AO, F_r, F_q, max_pin_burnup) on a fixed set of loading patterns per variant type.
5. **Blind probe.** Score those same patterns through the **current champion** (no fine-tune) and measure blind Spearman + MAE per target vs MASTER truth.
6. **Acceptance bands** (tied to the current blind-transfer levels in `data/curriculum/transfer_curve.json` — the median healthy blind cell): 
   - `f_r` Spearman **≥ 0.70** and MAE ≲ 0.5;
   - `cyclen` Spearman **≥ 0.60** and MAE ≲ 15 EFPD;
   - `cbc_max` Spearman **≥ 0.60** and MAE ≲ 50 ppm.
   (These are the record's typical blind levels; the ring-0 seed cell 5.25-5.5_f117 at f_r Sp 0.036 is the known cold-start outlier and is excluded as the floor reference.)
7. **Decision.** A variant that clears all bands is transfer-safe and may enter the optimizer with the 4b guard active. A variant that fails → **quarantine the geometry axis** (do not feed geometry-varied types to acquisition) until a **geometry-aware retrain**: add `pin_pitch`/`r_pellet`/`v_mod_over_v_fuel` as explicit design + featurizer axes and retrain the ensemble on a geometry-augmented population. Note this grows the channel inventory → `_assert_channel_parity`/`EncoderChannelMismatch` (`model_api.py:100, 291`) will (correctly) reject fine-tuning and force a **from-scratch retrain** — budget for a full v5 train, not a wave update. Expect `f_r`/`max_pin_burnup` to be the hardest targets to recover (section 2.1) and require the widest geometry sampling.

---

## Bottom line

- **Viable, with hard scoping.** Pin-level geometry variation is MASTER-compatible with **no core-model regeneration and no hard blocker**, *provided* the assembly-pitch token (20.7772) and 16×16/guide-tube layout stay frozen. The existing per-type design chain already regenerates XS/HFF/HGC.
- **Pin pitch is asymmetric:** +1.06 % absolute ceiling (water-gap closure), wide floor. Any ±pitch grid must respect this — ±3 % is not admissible upward.
- **Radius is intrinsically loud** (u_mass canary, +0.5 % → z≈2), so radius moves self-flag. **Pitch below ~2 % is the silent-but-wrong danger** — caught only by the spectral z-guard (xs_s12 canary) and the DeCART blind-probe, never by ensemble variance.
- **Model can propagate reactivity/lifetime direction through the harvested features but cannot disentangle moderation from the collinear enrichment/Gd manifold, and cannot reconstruct the reshaped pin form function** (F_r/pin-burnup least trustworthy). Treat any geometry campaign as gated by section 4c until a geometry-aware v5 retrain.
