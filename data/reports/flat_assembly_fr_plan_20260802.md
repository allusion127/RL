# Flat-assembly → core F_r: executable experiment plan

**Date** 2026-08-02 · **Author** analysis agent (TASK E) · **Status** ready to launch
**Preceding work** TASK A (surrogate API) · TASK B (design→library→core chain audit) ·
TASK C (core decomposition, FF targets) · TASK D (design-space sampling)

---

## 0. TL;DR

1. **The experiment that answers the user's question is launchable TODAY with zero DeCART and
   zero library work.** Three fuel sets that already exist as real ga80 lattices — `E1_E2`
   (control, FF 1.146/1.152), `E3_E4` (flatter, 1.101/1.139), `A8_A2` (peakier, 1.157/1.178) —
   all resolve at `fallback_level 0` (native equilibrium restart) inside
   `3_GA_Surrogate/FEASIBLE_PACKAGE`. Running the **same loading pattern** through all three is a
   3-point dose–response test of `F_r = A · p_node · FF` with a *predicted slope of 1.320*.
   Cost: 3 equilibrium chains (~10-35 MASTER cycles), no new assets.
2. **The DeCART spend is justified by ONE new fact found while writing this plan**: the 24 frozen
   `dec_FA` templates carry *bad Gd layouts*. At identical knobs (u_high 5.00, gd_wt 6, n_gd 20,
   z1/PB) the template layout `2:2;5:2;6:4` gives **FF 1.1659** while the ga80-style layout
   `1:1;4:1;6:4` gives **FF 1.1166** — **ΔFF 0.049 ⇒ ΔF_r 0.065**. Enrichment tuning inside the
   buildable set is worth ~0.002; *layout* is worth ~0.05. The payoff of new lattices is entirely
   in hand-authoring 3 new Gd pin maps.
3. **Recommended spend: 4 new lattices = 1 DeCART wave = ~13 min wall**, forming **two** new
   pairs — `X1_X2` (reactivity-matched to E1_E2, isolates FF) and `X3_X4` (flat extreme, sizes
   the prize). Predicted fixed-LP core F_r: **1.510** and **1.454** vs the control's 1.521.
4. **Honest limit**: at *matched reactivity* the maximum FF gain is −0.031 (E1 role) and only
   −0.009 (E2 role, which is the hot slot in this pattern). The large −0.067 F_r gain of `X3_X4`
   is bought with reactivity, i.e. with boron and cycle length — it is **not** a free lunch and
   the experiment's real job is to measure that trade, not to confirm a foregone conclusion.

---

## 1. The design set to realize (item 1)

### 1.1 Why 4, and why these 4

* A **pair** is the unit of loadability in this program (`bases/<A_B>`, `CaseKey(pair, feed)`).
  A meaningful core test therefore needs an even number of new types, minimum 2.
* DeCART2D runs **4-way parallel** on 104 at 740–780 s/type. **N=4 costs the same wall clock as
  N=2** (one 13-minute wave). 4 is the free upgrade from "one pair" to "two pairs".
* Two pairs is exactly what the experiment needs: one **reactivity-matched** pair (clean
  attribution, small effect) and one **flat-extreme** pair (confounded, large effect). One pair
  alone cannot separate "flatness helped" from "reactivity helped".
* 8 types would give a third pair but doubles wall, disk (~35 MB/type in `hgc/` + 7.4 MB in
  `lib/`) and, more importantly, pushes the paramA library from 37 to 41 fuel COMP — past the
  40-COMP TotalBatcher ceiling probe that is the only evidence we have. **Do not exceed 4.**

### 1.2 The four designs

All share `gd_u = 4.0` (the value the chain hard-freezes, `spec.GD_CARRIER_ENR`), pattern
**PB / zoning z1**, and `du = 0.15 · u_high` (the 0.85 zoning-ratio compliance rule,
`compliance.py: ZONE_RATIO_TARGET`). FF is full 8-member ensemble, surrogate, CPU.

| id | role | u_high | u_low | gd_wt | n_gd | Gd positions (1/8) | **FF** | k(0) | rbar_eoc | rbar_peak | CBC_pred |
|----|------|--------|-------|-------|------|--------------------|--------|------|----------|-----------|----------|
| **X1** | E1-role (68 fresh slots) | 5.00 | 4.2500 | 6 | 20 | `1:1;4:1;6:4` | **1.1166** | 1.1060 | −0.09036 | +0.02945 | 904 |
| **X2** | E2-role (53 fresh slots) | 5.00 | 4.2500 | 10 | 24 | `1:1;4:1;5:5;6:3` | **1.1441** | 1.0451 | −0.09459 | +0.00942 | 380 |
| **X3** | flat, E3-layout anchor | 5.00 | 4.2500 | 8 | 16 | `1:1;5:2;5:5` | **1.1049** | 1.1335 | −0.08919 | +0.03751 | 1115 |
| **X4** | TASK-D boron-limited optimum | 5.25 | 4.4625 | 6 | 16 | `1:1;5:2;5:5` | **1.1012** | 1.1557 | −0.07436 | +0.05227 | 1501 |

Reference points (measured from real HGC `%DIST`, TASK C):
`E1` 1.146 / `E2` 1.152 / `E3` 1.101 / `E4` 1.139 / `A8` 1.157 / `A2` 1.178 /
`Q1` 1.122 / `Q2` 1.174 / `Q7` 1.205 / `Q8` 1.209.

**Reactivity match quality** (surrogate 3-batch reactivity, `B_c = 24.7327 MWD/kgHM`):

| | rbar_eoc | rbar_peak |
|---|---|---|
| ga80 E1 | −0.08742 | +0.03185 |
| **X1** | −0.09036 | +0.02945 |
| ga80 E2 | −0.09312 | +0.00925 |
| **X2** | −0.09459 | +0.00942 |

X2 matches E2 to 150 pcm / 17 pcm — better than the ±290 pcm sd of the screen constant itself.
X1 matches E1 to ~290 pcm. **`X1_X2` is therefore a like-for-like replacement of `E1_E2`**:
same reactivity balance, flatter pin maps.

`X3_X4` is deliberately *not* matched — it is the flattest pair the boron gate allows, and it
carries ~+900 pcm of extra pair reactivity. That is the point of having both arms.

### 1.3 What was rejected, and why (this is the honest part)

* **The buildable-today set loses.** Restricting to the 4 frozen template layouts
  (`2:2;6:4` / `3:1;6:4` / `2:2;5:2;6:4` / `2:2;4:1;5:5;6:3`) and enumerating all 984 legal
  (u_high × gd_wt × n_gd × z) combinations at full ensemble: **at matched reactivity the best
  realizable design is *worse* than the ga80 type it would replace** — 1.1658 vs E1's 1.146,
  1.1753 vs E2's 1.152. A zero-template-work DeCART campaign would make F_r **worse**. This is
  the single most important negative result in the plan.
* Opening the layout (all 89 legal placements at n_gd 12/16/20/24, 5,874 designs) flips it:
  1.1151 vs E1's 1.146 (−0.031) and 1.1432 vs E2's 1.152 (−0.009).
* `gd_u` is a non-issue: 4.3 → 4.0 costs +0.0011 … +0.0019 FF, below the surrogate's own
  ~0.002 resolution floor. **Keep gd_u = 4.0** and the chain's hard freeze never binds.
* n_gd ≤ 12 designs (the flattest in raw FF) all fail the CBC ≤ 1550 ppm gate — TASK D's
  finding, re-confirmed here. Not sampled.

---

## 2. Ordered command list (item 2)

Absolute root: `C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산`
(abbreviated `<ROOT>` below; every command below quotes the real path).

### TIER 0 — no DeCART, no library, launchable now (~1–2 h MASTER)

**T0-1 (free, seconds) — preflight, already verified by this agent:**

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; python fr_arms.py --list
```

Confirms: reference record `b0ff11ef16de` (E1_E2, feed 121, F_r 1.5207, node_peak 1.2085),
`library_dims (nbatch, ncomp) = (83, 85)`, and that `E1_E2` / `E3_E4` / `A8_A2` all resolve at
`fallback_level 0` with native restarts.

**T0-2 (EXPENSIVE — 3 equilibrium chains) — the control + two dose points:**

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; python fr_arms.py --arm A0 --arm A1 --arm A2 --package ../3_GA_Surrogate/FEASIBLE_PACKAGE --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe --run-dir runs/fr_arms_t0 --workers 12 --max-cycles 16 2>&1 | Tee-Object -FilePath runs\fr_arms_t0.log
```

On a fleet box the same command runs from the kit dir with
`--package FEASIBLE_PACKAGE --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe`; the kit ships no
`records.parquet`, which is why `fr_arms.py` falls back to the portable
`5_RL/fr_arms_pattern.txt` (618-char packed pattern, written alongside the script). Ship
`fr_arms.py` + `fr_arms_pattern.txt` into the kit and nothing else is needed.

**T0-3 (EXPENSIVE, 1 chain) — reproducibility floor.** Re-run A0 from the same restart into a
different run dir. Everything downstream needs to know whether a 0.010 F_r difference is
resolvable:

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; python fr_arms.py --arm A0 --package ../3_GA_Surrogate/FEASIBLE_PACKAGE --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe --run-dir runs/fr_arms_t0_rep --workers 12
```

Results land in `runs/fr_arms_t0*/fr_arms_results.jsonl` (arm, pair, status, n_cycles,
`fom` = F_r/CBC_max/F_q/cyclen, node_peak). Nothing is written into `data/`.

### TIER 1 — the 4 new lattices (gate on Tier 0, see §3.4)

**T1-1 (MANUAL, no code exists for this — the real cost) — author 3 new `dec_FA` pin maps.**
`lattice.edit_dec_text` edits only `UO2 92235`, `UO2_2 92235`, `UO2G 6408`; **Gd pin positions
come from the template and nothing in the repo writes them**. Copy the matching base template and
move the Gd pin ids:

| new layout | n_gd | base template to copy |
|---|---|---|
| `1:1;4:1;6:4` | 20 | `<ROOT>\0_APR1400\5.8_5.1\FA\IGD_20\6_20_z1\dec_FA_*.inp` |
| `1:1;4:1;5:5;6:3` | 24 | `<ROOT>\0_APR1400\260624\FA\IGD_24\10_24_z1\dec_FA_*.inp` |
| `1:1;5:2;5:5` | 16 | `<ROOT>\0_APR1400\5.8_5.1\FA\IGD_16\8_16_z1\dec_FA_*.inp` (also serves X4 via `6_16_z1`) |

Ground truth for the 1/8 → full-16×16 expansion is `surrogate/features.py:85-99`
(`quarter[i,j] = lower_triangle[max(i,j)][min(i,j)]`, mirrored 4×). Guide tubes are fixed at 1/8
`(0,0),(3,3),(4,3),(4,4)` and must not move. **Verify each authored deck before running DeCART**
by re-extracting the Gd census the same way TASK C recovered ga80's layouts from `%DIST`.
Note `1:1;5:2;5:5` is *exactly ga80 E3's* layout — a layout already proven manufacturable and
already run through DeCART once, which is why X3 doubles as the surrogate validation anchor.

**T1-2 (free, seconds) — snapshot the rollback state.** `MAS_XSL`/`MAS_HFF` keep exactly ONE
`.bak` generation; a second consecutive rebuild destroys the only rollback copy.

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; Copy-Item data\design\package\lib data\design\package\lib.snap_20260802 -Recurse; Copy-Item data\design\package\designs.json data\design\package\designs.json.snap_20260802; Copy-Item data\design\package\registry.json data\design\package\registry.json.snap_20260802
```

**T1-3 (EXPENSIVE — 1 DeCART wave, ~13 min, 104 only) — lattices + library + manifest + ingest.**
Use the audited driver from TASK B (`5_RL/realize_designs.py`), with `NEW = [...]` set to the four
`FuelDesign` objects and `run_batch` pointed at the hand-authored template dirs. DeCART2D exists
only on 104 (`D:\DeCART_MASTER\BIN\decart2d1.1m5omp.exe`) and `resolve_template` needs the
`0_APR1400` tree, which the fleet kits do not carry.

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; python realize_designs.py 2>&1 | Tee-Object -FilePath realize_designs_20260802.log
```

The driver: registers the 4 designs into the **package** registry (never the store registry —
they have diverged, 4 vs 33 aliases), runs DeCART 4-way, rebuilds `MAS_XSL`/`MAS_HFF` over the
**union** of 33 old + 4 new HGCs (`build_master_library` refuses a partial request), rewrites
`designs.json`, and re-ingests `fuel_types.parquet`. Expect `ncomp = 42`, `nbatch = 40`.

**Record the auto-assigned aliases** (the registry assigns the next free 2-char ids, most likely
`T3`–`T6`; it will NOT hand out `X1`–`X4`) and update the `B2`/`B3` entries in `fr_arms.py`
accordingly.

**T1-4 (free, seconds) — verify the rebuild and enumerate what just went stale:**

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; $n=(Select-String -Path 'data\design\package\lib\MAS_XSL' -Pattern '^COMP ' | Measure-Object).Count + 5; "current ncomp=$n"; Get-ChildItem data\design\package\cores -Recurse -Filter 'MAS_INP_cy*.inp' | ForEach-Object { $d=(Select-String -Path $_.FullName -Pattern '^\s+10\s+10\s+27\s+(\d+)\s+(\d+)').Matches[0].Groups[2].Value; '{0,-8} deck_ncomp={1} {2}' -f $_.Directory.Parent.Name, $d, $(if([int]$d -eq $n){'OK'}else{'STALE -> re-bootstrap'}) }
```

**T1-5 (EXPENSIVE — 4 × ~11 MASTER cycles ≈ 4 min each) — bootstrap the two new pairs AND
re-bootstrap the two paramA control pairs** (every existing restart is invalidated by the ncomp
38 → 42 shift; `P0_P1` was already stale at ncomp 16 before this campaign):

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; foreach ($p in @('<X1>_<X2>','<X3>_<X4>','Q1_Q2','Q7_Q8')) { python -m lpopt design bootstrap --input fill_198.inp --pair $p --feed 121 }
```

**T1-6 (free, seconds) — confirm loadability at fallback_level 0:**

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; python -c "from pathlib import Path; from lpopt.search.assets import CaseAssetResolver; from lpopt.search.resolver import paramA_registry_aliases, paramA_library_dims; from lpopt.vendor.masterrl.domain import CaseKey; pkg=Path('data/design/package').resolve(); al=paramA_registry_aliases(pkg); dims=paramA_library_dims(pkg,al); print('dims',dims); r=CaseAssetResolver(pkg, library_id='paramA', registry_aliases=al, library_dims=dims); [print(k, r.resolve(CaseKey(k,121)).fallback_level, r.resolve(CaseKey(k,121)).restart_provenance) for k in ('<X1>_<X2>','<X3>_<X4>','Q1_Q2','Q7_Q8')]"
```

**T1-7 (EXPENSIVE — 4 equilibrium chains) — the paramA arms, same pattern, same script:**

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; python fr_arms.py --arm B0 --arm B1 --arm B2 --arm B3 --package data/design/package --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe --run-dir runs/fr_arms_t1 --workers 12 2>&1 | Tee-Object -FilePath runs\fr_arms_t1.log
```

To run T1-7 on the fleet, ship the paramA package first
(`python -m lpopt export-produce-kit ...`); it is ~1.3 GB with 37 types.

### TIER 2 — only if Tier 1 confirms (re-optimize the LP)

A fixed pattern measures *transfer*, not *achievable optimum*. Flatter assemblies change the
k-inf distribution, hence `node_peak` **and** the amplification `A`, so the real answer needs a
search with the new types:

```powershell
cd "C:\Users\USER\Desktop\CT&RPL\2_Project\KNF_LEU+ 시범연료봉 장전 인허가 과제\2026\2_계산\5_RL"; python -m lpopt optimize --input fill_198.inp --budget 150
```

with `[model].library_id = "paramA"`, `[verify].package_root = "data/design/package"`, and — the
policy change this whole line of work exists to justify — `gate_noreg_fr_guard_enabled = True`
with the F_r target moved from the deferred 1.689 safety gate down to 1.50.

---

## 3. The core-side test (item 3)

### 3.1 Control arm and the attribution argument

**Control = arm A0**: the *identical* 69-card loading pattern of store record `b0ff11ef16de`
(pair `E1_E2`, feed 121, ga80), run from the *native* `bases/E1_E2/MAS_RST.APRQ_11_0635.19`
restart in the *same* package, with the *same* equilibrium protocol (`max_cycles 16`,
`consecutive 2`). The only thing that varies across arms is the identity of the two fresh batch
tokens. Every shuffle card, the feed count (121), the symmetry class (`rot61`) and the deck knobs
are byte-identical. A0 must reproduce **F_r = 1.5207, node_peak = 1.2085, CBC 1330.5 ppm,
cyclen 632.3 EFPD**; if it does not, the harness — not the physics — is what changed, and the
whole experiment is void.

This is the *same-loading-pattern-family* comparison the task asks for. It is stronger than
"family": it is the same pattern, card for card.

### 3.2 Predicted effect sizes

The decomposition is `F_r = A · max_slots( p_boc(slot) × FF_type(bu_slot) )` with the
in-core flux-gradient amplification `A = 1.0928` measured on this very core (TASK C). Verified
here against the stored maps: the decomposition reproduces the control at **1.5208 vs the
observed 1.5207**. The hot slot is a fresh "E2-role" slot at `p_boc = 1.2080` (quarter index 6
and 52); the hottest "E1-role" slot is `p_boc = 1.1934`.

| arm | fuel set | FF (E1-role / E2-role) | **predicted F_r** | Δ vs control | attribution |
|-----|----------|------------------------|-------------------|--------------|-------------|
| **A0** | E1_E2 (control) | 1.146 / 1.152 | **1.5208** (obs 1.5207) | 0 | — |
| **A1** | E3_E4 | 1.101 / 1.139 | **1.5036** | −0.017 | confounded (+870 pcm pair reactivity) |
| **A2** | A8_A2 | 1.157 / 1.178 | **1.5551** | +0.034 | negative control |
| **B0** | Q1_Q2 (paramA) | 1.122 / 1.174 | **1.5498** | +0.029 | dose point |
| **B1** | Q7_Q8 (paramA) | 1.205 / 1.209 | **1.5960** | +0.075 | dose point (peaky end) |
| **B2** | **X1_X2 (new)** | 1.117 / 1.144 | **1.5103** | **−0.010** | **reactivity-matched ⇒ clean** |
| **B3** | **X3_X4 (new)** | 1.105 / 1.101 | **1.4538** | **−0.067** | confounded (flat *and* reactive) |

Predicted regression across all arms: `F_r = 1.3201 · FF_hot` (slope = `A × p_hot` =
1.0928 × 1.2080), zero intercept, `R² → 1` if the separable decomposition holds.

### 3.3 What FALSIFIES the hypothesis

The hypothesis — *flatter assemblies lower core F_r for the same loading pattern* — is falsified
if **any** of the following holds after Tier 0:

* **F1 — no transfer.** The fitted slope of measured `F_r` on lattice `FF_hot` across
  {A0, A1, A2} is below **0.66** (half the predicted 1.320) or is not significantly positive.
  That would mean MASTER's pin reconstruction does not carry the lattice form function into
  `F_r`, and assembly-level FF optimization is the wrong lever entirely.
* **F2 — wrong sign.** A1 (flatter fuel) measures `F_r ≥` A0, or A2 (peakier fuel) measures
  `F_r ≤` A0.
* **F3 — not resolvable.** The A0-vs-A0-replicate spread (T0-3) exceeds **0.005** in F_r. Then
  the reactivity-matched Tier-1 effect (−0.010) is inside the noise and the *matched* arm B2 can
  never be decisive — only the confounded B3 could be, and it proves less.
* **F4 — the trade eats the gain.** B3 delivers its −0.067 F_r but breaches `CBC_max > 1550 ppm`
  (the hard `acq.cbc_limit`) or `F_q > 2.41`. The control already sits at 1330 ppm and X3_X4
  carries ~+900 pcm more pair reactivity, so this is a live risk, not a formality. If it fires,
  the correct conclusion is TASK D's: *F_r 1.45 is a boron problem, not a pin-map problem.*
* **F5 — the amplification moves.** If measured `A = F_r / max(p_boc × FF)` shifts by more than
  its flat-regime spread (1.024–1.152) between arms, then `A` is fuel-dependent, the separable
  model is invalid, and the whole "lattice FF target" framing (TASK C's 1.174/1.136/1.098) must
  be rebuilt.

**Confirmation** requires: slope within [1.0, 1.6], A1 < A0 < A2 in F_r, B2 < A0 by ≥ 0.005 with
CBC/cyclen within 1σ of the control, and A stable to ±0.03 across arms.

### 3.4 Gate between tiers

Do **not** spend the DeCART wave until Tier 0 returns. If F1/F2/F3 fires, the correct next move
is not new lattices but the amplification factor `A` (spans 1.024–1.152 across flat cores; the
store's lowest-F_r core wins with `A = 1.033`, *not* by being flat) — worth ~12% on F_r versus
the ~1% left in FF at matched reactivity.

---

## 4. Risks, blockers, mitigations (item 4)

### 4.1 BLOCKER — the buildable design space cannot beat ga80 (severity: campaign-killing)

Full-ensemble enumeration of all 984 legal designs on the 4 frozen template layouts shows that at
matched reactivity **every** realizable design is *peakier* than the ga80 type it would replace
(1.1658 vs E1's 1.146; 1.1753 vs E2's 1.152). Running DeCART on template-layout designs would
spend 13 minutes to make F_r worse.
**Mitigation:** hand-author 3 new `dec_FA` pin maps (T1-1). This is manual work with no code
support and it is the only load-bearing manual step in the plan. Verify each authored deck by
re-extracting the Gd census from the produced HGC `%DIST` before trusting the result.

### 4.2 DeCART availability and licence

`decart2d1.1m5omp.exe`, `master4.0m4_r1.exe`, `prolog41m4.exe`, `TotalBatcher4.exe`, `MAS_REF`
and all 24 template dirs are **present and verified on 104**. The fleet boxes (181/198/199) carry
MASTER but their DeCART2D presence is unverified *and moot* — their kits ship only
`FEASIBLE_PACKAGE`, not the `0_APR1400` template tree `resolve_template` requires.
**Lattice production is a 104-only, serialized, 13-minute step.** No licence-server dependency is
recorded anywhere in the chain; the binaries are invoked directly by `subprocess`. If DeCART is
unavailable on 104 the entire Tier 1 is blocked — see §5.

### 4.3 COMP-count limit

`ncomp = 5 + N_fuel`, `nbatch = 3 + N_fuel`, read live from `lib/MAS_XSL`. Today paramA is 33
types (ncomp 38, the largest ncomp with *actual MASTER evidence*). +4 → **ncomp 42**. The
TotalBatcher ceiling probe passed at 40 fuel COMP (ncomp 45) but **was never driven through
MASTER**. **Mitigation:** T1-5 bootstraps one pair first — that *is* the cheap MASTER smoke at
ncomp 42. If it fails, fall back to a 2-type campaign (X3 + X4 only, ncomp 40).
Hard cap the package at 100 types forever: past 100 the alias pool wraps `Z9 → A0`, which sorts
*before* `P0`, silently reordering COMP indices and invalidating every restart with no error.

### 4.4 Library rebuild invalidates every restart

Adding 4 types shifts `%GEN_DIM` for **all** cases; `validate_reload_deck` refuses any deck whose
dims differ from the live library. Budget **4 bootstraps** (2 new pairs + Q1_Q2 + Q7_Q8), ~4 min
each. `P0_P1` is already stale (ncomp 16 vs 38) and stays stale — it is not used here.
`build_master_library` also refuses a partial request, so always rebuild over the union
(the driver does). Snapshot `lib/` first (T1-2): only one `.bak` generation survives.

### 4.5 Surrogate domain of validity at the chosen designs

| axis | chosen | trained | verdict |
|---|---|---|---|
| n_gd | 16, 20, 24 | 0,4,8,12,16,20,24 | **interior** — no extrapolation (n_gd 28 is a hard wall, +1750 pcm) |
| gd_wt | 6, 8, 10 | 6–10 | **X2 and X4 sit on a bound** (10 and 6) |
| u_high | 5.00, 5.25 | 5.00–7.00 | **X1/X2/X3 sit on the lower bound 5.00** |
| gd_u | 4.0 | 3.00–4.30 | interior |
| **du** | **0.750, 0.7875** | **0.4–0.8** | **X4 at 0.7875 is inside; X1–X3 at 0.750 inside — but off the 0.1 grid** |
| pattern | PB | PA/PB | interior |

**Flags.** (a) Three of four designs sit on `u_high = 5.00`, the lower bound — the flat corner
genuinely lives there, but bound-stacking is where TASK A measured elevated k error. X2 stacks
*two* bounds (`u_high 5.00` × `gd_wt 10`). (b) `du = 0.15·u_high` is **off the 0.1 validation
grid** by construction — the compliance rule and the surrogate grid are incompatible. TASK D
validated this on 24 template types (FF error mean −0.0001, sd 0.0007, max 0.0013) and TASK C
saw agreement hold at du up to 0.99, so the pin-map head extrapolates in du without visible
damage; the **k-curve head has no such evidence**. (c) The surrogate exposes **no uncertainty**;
the error bars used throughout are residual statistics. (d) FF differences below **0.002** are
noise — X3 (1.1049) vs X4 (1.1012) is *not* a resolvable ranking, and neither is X4 vs ga80 E3
(1.101). (e) FF bias is **conservative** (+0.0002…+0.0012 over-predicted), so a design the
surrogate calls flat is, if anything, flatter.
**X3 is the mitigation**: its layout is exactly ga80 E3's, so its DeCART result is a direct
surrogate-vs-truth check (predicted 1.1049 at these knobs; E3 measured 1.101 at *its* knobs).
Do not skip X3 to save a slot.

### 4.6 Library-id plumbing (quantified)

**Cost is ZERO if the new types go into the existing paramA package**, which this plan does:
`ingest_fuel_types` writes them as `library_id = "paramA"` rows in `fuel_types.parquet`, the
resolver reads the package registry, and `[model].library_id`/`[verify].package_root` already
have working values (`fill_198.inp`).

If instead a **new** library id were minted (e.g. building a 36-ga80 + 4-new hybrid so the ga80
control and the new types share one `MAS_XSL`), the cost is:

* 3 config fields (`[model].library_id`, `[verify].package_root`, `[curriculum].library`) — trivial;
* `FuelLibrary` rows under the new id — but **the ga80 package ships 36 HGCs against an 80-COMP
  `MAS_XSL` and ZERO `.out` files**, so (i) the ga80 library can never be rebuilt at all, and
  (ii) `ingest_fuel_types` cannot produce rows for ga80 types (it reads `designs.json` + `FA_*.out`,
  and ga80 has neither);
* every `bases/` restart in `FEASIBLE_PACKAGE` (21 pairs) becomes invalid because the COMP index
  set changes from 80 to 40 → 21 re-bootstraps, and `design bootstrap` cannot bootstrap ga80 pairs
  (no `DesignRegistry` records) → **new code required**;
* the trained PosValNet ensemble is library-conditioned (`featurize` uses each row's own
  `library_id` provenance) → any RL/`optimize` use of the new id needs retraining.

**Conclusion: do not mint a new library id.** Extend paramA (cost 0) and accept that the ga80
control arm (Tier 0) and the paramA arms (Tier 1) live in different libraries — which is fine,
because each tier carries its *own* internal control (A0 for ga80, B0/B1 for paramA).

### 4.7 Disk and wall clock on 104

| step | wall | disk |
|---|---|---|
| T1-3 DeCART ×4 (4-way parallel) | ~13 min | 4 × 35 MB in `hgc/` + 4 × 7.4 MB in `lib/` ≈ 170 MB |
| T1-3 TotalBatcher (37 types) | ~1 min | `MAS_XSL` ~14.4 MB, `MAS_HFF` ~15.0 MB (+ one `.bak` each) |
| T1-5 bootstrap ×4 | ~16 min | 4 × ~6.6 MB `MAS_RST` |
| T0-2/T0-3 (4 chains) | 20–70 min | run dirs, purged between cycles |
| T1-7 (4 chains) | 20–70 min | idem |
| **total** | **~2–3 h** | **< 0.3 GB** |

### 4.8 Other recorded hazards carried forward

* Never use `lpopt design run` for these designs — it re-derives `e2` from a 1-decimal `type_id`
  and silently loses the second decimal (X4's `u_low = 4.4625` would become 4.5). The driver
  script passes `FuelDesign` objects directly.
* `data/design/decks/` is stale and disagrees with the packaged HGCs; the authoritative Gd layout
  is always the HGC `%DIST` map.
* Two divergent registries exist (`data/design/registry.json`, 4 aliases vs
  `data/design/package/registry.json`, 33). Register into the **package** one only.
* Surrogate k against legacy decks carries a **−2396 pcm** offset (xenon TR vs EQ, UO2G density
  10.01 vs the trained convention). All reactivity comparisons in this plan are done in
  *surrogate* space with ga80 values shifted by −0.024 in ρ, which cancels it.
* Everything the surrogate produces is a single-assembly **infinite-lattice** quantity at 500 ppm
  boron with equilibrium xenon. The separable `F_r = A · p · FF` model is an approximation; §3.3
  F5 is the test of that approximation.
* The local GTX 1080 Ti (sm_61) cannot run the surrogate — `--device cpu` is mandatory on 104.

---

## 5. What cannot run here, and the partial fallback (item 5)

Everything in this plan **can** run on the machines available:

* DeCART2D — **present on 104**, verified.
* MASTER — present on 104 and on 181/198/199.
* TotalBatcher/PROLOG/MAS_REF and all 24 `dec_FA` templates — present on 104.
* The three Tier-0 fuel sets — present with native restarts, verified to resolve at level 0.

Two things are **not** available, with fallbacks:

1. **The ga80 library can never be rebuilt.** `FEASIBLE_PACKAGE/hgc` has 36 HGCs against an
   80-COMP `MAS_XSL` and no `.out` files. New assembly types therefore **cannot** be added to the
   ga80 library, and the flattest-core control and the new lattices **cannot** be placed in one
   library. *Fallback (adopted):* each tier carries its own internal control — A0 for ga80,
   B0/B1 for paramA — and the cross-tier comparison is made only through the FF→F_r regression,
   never as a raw F_r difference. If the 44 missing ga80 HGCs and the 36 `.out` files are
   recoverable from the original GA campaign, say so — that would allow a single-library design
   and remove this caveat entirely.
2. **`lpopt` cannot bootstrap or manifest ga80 types** (no `designs.json`, no `.out`, no design
   variables for 44 of 80). Any plan requiring a *new* ga80 pair needs new code.

**If DeCART2D turns out to be unusable on 104** (licence, template authoring blocked, or the
ncomp-42 smoke fails), Tier 0 alone still answers a large part of the user's question: it measures
whether lattice FF transfers into core F_r at fixed LP, with a 3-point dose–response spanning
FF 1.101 → 1.178 and a predicted F_r span of 1.504 → 1.555. What it *cannot* do without new
lattices is separate flatness from reactivity (A1 is confounded by +870 pcm), and it cannot reach
the flat extreme (predicted F_r 1.454) because no existing pair is flat on both members.
A weaker but zero-cost substitute for the flat extreme is to run the same pattern with the
flattest available *paramA* pair (B0 = `Q1_Q2`, FF 1.122/1.174) — but note its predicted F_r is
**1.550, worse than the control**, because its E2-role member is peaky. That asymmetry is itself
the finding: **the hot slot's assembly is the only one that matters, and no existing library has
a flat *low-reactivity* assembly.**

---

## 6. Artifacts produced with this plan

| file | contents |
|---|---|
| `5_RL/fr_arms.py` | the arm driver — `--list` (free) and `--arm` (MASTER); writes `runs/<dir>/fr_arms_results.jsonl` |
| `5_RL/fr_arms_pattern.txt` | portable 618-char packed reference pattern (record `b0ff11ef16de`) so fleet kits need no store |
| scratchpad `realizable984_ens.csv` | full-ensemble scoring of all 984 buildable-today designs |
| scratchpad `matched_full.csv` | 5,874 designs over all 89 legal Gd placements, single-member FF + 3-batch reactivity |
| scratchpad `final4.py` | full-ensemble scoring of X1–X4 (+ gd_u and template-layout contrasts) |

**First command to run:** T0-1 (free), then T0-2.
