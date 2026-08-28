# (c) Cyclen-isolated dual-trunk — design DRAFT (awaiting user decision; NOT implemented)

Status: **draft only.** Do not implement without explicit user approval.

## 1. Problem this solves
The honest-gate history established that the frontier/low-band data irreducibly re-shapes the **shared trunk** representation, which lifts boundary-F_r ranking but breaks high-band cyclen ranking. Four levers failed to reconcile them:

| attempt | lever | cyclen worst-drop | boundary F_r ρ |
|---|---|---|---|
| #4/#5a/#5b | loss protection (rank↑, distill pin) | 0.40-0.46 (FAIL) | 0.41-0.45 |
| #6 | data re-anchor (doubling) | 0.405 (FAIL) | 0.41 |
| **#7 (promoted)** | **freeze trunk + finetune F_r/node_peak heads** | **0.013 (PASS)** | **0.322** |

#7 (the promoted champion) buys cyclen safety by freezing the shared trunk — but that caps boundary-F_r recovery at **0.322** (only +0.03 over champion 0.29) because with the trunk frozen only the F_r head's linear reweighting of frozen features can adapt. The full **0.41** (+0.12) requires trunk adaptation, which is exactly what breaks cyclen.

**Goal of (c):** recover the full boundary-F_r headroom (~0.41) at **zero cyclen cost** (worst-drop ≈ 0), by giving F_r/node_peak a trunk that adapts freely while cyclen reads a protected trunk.

## 2. Core design — two disjoint representation paths
Split the network into two independent paths sharing (at most) the raw featurized input:

- **Cyclen path (FROZEN, = champion):** `stem_C → blocks_C → head_trunk_C → {cyclen mu, cyclen log_sigma, cyclen quantiles}` + the champion's cyclen cell-calibration + cyclen physics prior. Initialized verbatim from champion `20260721_105824` and fully frozen. Guarantees cyclen prediction is byte-identical to the champion → gate cyclen worst-drop = 0 by construction (same guarantee #7 already achieves, but now the F_r path is unshackled).
- **Main path (TRAINABLE):** `stem_M → blocks_M → head_trunk_M → {f_r mu/σ/quantile, cbc, ao, max_asm_bu} ; map_head_M (node_peak)`. Trained freely on the new corpus (frozen-init from champion OR fresh init). Full trunk adaptation → targets 0.41 boundary-F_r + the node_peak gains.

Serving combines: cyclen column from the frozen path, all other targets + maps from the main path.

**Sharing knob (the key design decision):**
- **Option A — fully disjoint** (no shared params): two independent stems+trunks. Strongest cyclen guarantee, ~2× the per-member param/compute of a single trunk. Cleanest.
- **Option B — shared FROZEN stem, diverging trainable/frozen blocks:** one `stem` (frozen from champion), then `blocks_C` (frozen) and `blocks_M` (trainable) branch off the shared frozen stem features. Saves the stem's params; cyclen still exact because everything on the cyclen path (frozen stem → frozen blocks → frozen heads) is champion. Main path adapts `blocks_M` + heads on frozen stem features — more adaptation than #7 (which froze the whole trunk) but less than fully-disjoint (stem fixed). **Middle ground; recommended starting point** — the empirical question is whether a frozen stem still leaves enough headroom to approach 0.41 (if not, fall back to A).
- **Option C — shared TRAINABLE stem:** rejected. A trainable shared stem feeds the frozen cyclen blocks; adapting it for F_r shifts the cyclen path's input → cyclen drift → defeats the purpose.

## 3. Training
- Init: cyclen path frozen-loaded from champion; main path frozen-loaded from champion (warm) or fresh (cold — likely better for F_r since it's escaping the champion's F_r basin).
- Loss routing: cyclen loss terms (mu/σ/quantile/rank) flow ONLY to the frozen cyclen path → they are no-ops (or drop them entirely). All F_r/node_peak/cbc/ao/map losses flow to the main path. Distillation: F_r-target distill on the main path as usual; cyclen distill unnecessary (frozen).
- Ensemble: same 5-member deep ensemble; each member's cyclen path = corresponding champion member (frozen), main path trained independently.
- Gate: unchanged honest protocol. Expected: cyclen worst-drop ≈ 0 (frozen); boundary-F_r ρ target ~0.41; node_peak retained/improved.

## 4. Implementation surface (for scoping — NOT to build yet)
1. `net.py`: new `PosValNetDualTrunk` (or a `dual_trunk` config path on `PosValNet`) with the two paths + a serve-time combiner. Backward-compatible: the flag-off net stays byte-identical (the whole existing v5 contract).
2. `train.py`: init both paths from champion members; route losses; freeze the cyclen path (reuse the `--freeze-trunk-cyclen` param-group machinery, extended to whole-path freeze); a `--dual-trunk[=A|B]` flag.
3. `model_api.py`: serve-time combine (cyclen from path C, rest from path M) — must keep `predict`, `predict_map_peak`, calibration, and the state_dict layout consistent for the gate + acquisition.
4. Calibration/prior: copy champion cyclen calibration + physics prior verbatim (as #7 does); re-fit F_r calibration on the main path.
5. Tests: cyclen path byte-frozen (already have the pattern from `test_freeze_finetune`); main path adapts; serve combiner picks the right columns; flag-off byte-identity; full suite green.
6. Retrain on GPU 0 (freeze-heavy → still fast) → gate → promote on pass.

**Cost vs #7:** materially larger — a new net topology + serve combiner + training routing + tests, ~2× per-member compute (Option A) or ~1.3-1.6× (Option B), plus one retrain+gate cycle. #7's freeze was a ~50-line change; this is a new architecture.

## 5. Recommendation
#7 is promoted and banks a safe, honest gain now. Pursue (c) **only if** the extra boundary-F_r headroom (0.32 → ~0.41) is worth the architecture investment. Suggested path if approved: implement **Option B** first (shared frozen stem — lowest cost), gate it; if its F_r recovery still falls short of ~0.41, escalate to **Option A** (fully disjoint). Either way the cyclen guarantee is exact (frozen path).
