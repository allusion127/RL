# Inference backend: local CPU vs remote GPU (plan 4.7)

Makes the campaign's screening inference backend user-selectable from the deck and
benchmarks both paths on a real case.

**Deck knob** — under `[model]`:

```toml
[model]
device    = "cpu"          # local torch device for the CPU path + wave fine-tune (unchanged)
inference = "local_cpu"    # "local_cpu" (default) | "remote_gpu"
```

* `inference = "local_cpu"` — score every candidate on this PC's CPU. Network-
  independent; a campaign survives a server outage. This is the default when the
  key is **omitted** (an omitted key defers to the legacy `remote_screening`, which
  is `false`, so an unchanged deck behaves exactly as before).
* `inference = "remote_gpu"` — offload the large-pool screen/deepen bulk inference
  to `gpu2-6000` **GPU 1** (the `[remote]` table), returning only the ranked score
  arrays (a few MB). It probes the server first (5 s) and — on an unreachable
  server **or any per-batch transport error mid-campaign** — logs loudly and falls
  back to local CPU rather than aborting (never partial scores).
* `inference` takes precedence over the legacy `remote_screening` / `remote_screening_min`
  keys (both still work; `remote_screening_min` = the batch size below which a batch
  stays on local CPU because the ssh round-trip is not worth it, default 5000).

The wire path (`lpopt.model.remote_infer` + `lpopt.remote.remote_infer`) ships the
compact **packed patterns** (`Pattern.canonical()`, ~0.6 KB each) + per-item
`CaseKey`, re-encodes + runs the ensemble on the GPU, and returns the **raw**
ensemble arrays exactly as the local choke point produces them, so the campaign's
denorm + calibration stack is reused unchanged and the reassembled object is
identical to the local one.

---

## Case

* **Core-average enrichment 5.5 w/o, feed 117** — curriculum cell `5.5-5.75_f117`,
  pairs `G3_G4`, `H1_H2`, `H3_H4` (from `data/curriculum/state.json`). All three
  pairs land at e_core 5.500 at split 0.5.
* **Champion**: `data/models/20260721_061913` (resolved from `state.json`
  `champion_model_dir`) — 5-member deep ensemble, cond_schema **v4**, 7 targets
  (`f_r, f_q, cbc_max, cyclen, ao_abs, discharge_burnup, max_pin_burnup`).
* **Pool**: 2,000 and 20,000 **unique, geometry-validated** candidates generated
  with the SAME campaign generators (`lpopt.search.construct.build_pool`), cached to
  disk so the local-CPU and remote-GPU paths score the **identical** pool.
* Measured on the local PC (24 threads) and the remote GPU 1 (RTX PRO 6000 Blackwell
  Max-Q) while GPU 1 was **idle** (v5 A/B training had finished). Pinned via the
  deck's `[remote] gpu = 1` — GPU 0 was never touched.

---

## Benchmark: predictions/s, both paths

| N | Local CPU wall | Local pred/s | Remote GPU wall (end-to-end) | Remote pred/s (e2e) | Remote wall (steady-state¹) | Remote pred/s (steady) | e2e speedup | steady speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000  | 13.45 s  | **148.7** | 9.43 s  | **212.1** | 5.69 s  | 351.6 | 1.43× | 2.36× |
| 20,000 | 193.22 s | **103.5** | 55.22 s | **362.2** | 47.30 s | 422.9 | **3.50×** | 4.09× |

¹ *steady-state* = one-shot ssh/python-startup (`ssh_startup`) and the one-time model
load excluded — i.e. the per-batch cost a long-lived screen amortizes. The campaign
prewarms the whole screen in ONE batch, so it pays the fixed overhead once and sees
the steady-state number.

Local CPU per-pattern cost **rises** with N (148.7 → 103.5 pred/s from 2 k → 20 k):
the encoded feature tensor is `float32[N, 26, 19, 19]` (≈ 0.75 GB at 20 k) and memory
bandwidth throttles the 24-thread CPU. The remote path speeds **up** with N because
its fixed overhead amortizes.

### Remote path decomposition (transfer / featurize / forward / return)

| N | transfer up | **featurize** (encode) | **forward** (GPU, 5 members) | return | model load (one-time) | ssh/py startup (one-time) | peak GPU mem |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000  | 0.50 s | 4.28 s  | 0.56 s | 0.35 s | 0.30 s | 3.44 s | 1,787 MB |
| 20,000 | 0.46 s | **43.27 s** | **3.12 s** | 0.45 s | 0.31 s | 7.61 s | 4,407 MB |

### Local path decomposition (same pool, for comparison)

| N | featurize (encode) | forward (CPU, 5 members) |
|---:|---:|---:|
| 2,000  | 2.07 s  | 9.95 s |
| 20,000 | 21.23 s | **152.16 s** |

### Is the GPU worth it — or does featurization dominate? **Both.**

* **The GPU is decisively worth it for the forward.** The ensemble forward collapses
  from **152 s on the local CPU to 3.1 s on the GPU at 20 k — a ~49× speedup** (≈18×
  at 2 k). This is the entire reason the remote path wins 3.5× end-to-end.
* **After that collapse, featurization dominates the remote wall.** At 20 k the remote
  steady-state is 47.3 s of which **featurize = 43.3 s (91.5 %)** and forward = 3.1 s
  (6.6 %); transfer + return are < 1 s each. Featurization is a single-threaded
  Python/numpy encode loop (`encoder.encode` per pattern) and is actually **slower on
  the GPU box (43 s) than locally (21 s)** because its single-core CPU is slower — the
  GPU does not help it at all.
* **Conclusion.** Remote GPU is the right backend for large-pool screening (3.5–4×
  faster at campaign screen sizes), but the forward is no longer the bottleneck — the
  next win would come from parallelizing/vectorizing featurization, not a faster GPU.
* **Payload is tiny** (packed patterns, not features): transfer up + return together
  are < 1 s even at 20 k, confirming the "ship patterns, encode remotely" choice.
* **Good neighbour**: peak GPU memory 1.8 GB (2 k) / 4.4 GB (20 k) of 97.9 GB, forward
  chunked at 5,000.

### Crossover pool size

Remote end-to-end already beats local at 2,000 (1.43×). Fitting
`remote_e2e ≈ 4.34 s + 2.54 µs·N` against `local ≈ 6.7 µs·N` (small-N rate) gives a
crossover at **≈ 1,000 candidates** for a single one-shot remote call (which pays the
ssh + model-load overhead once). Below ~1,000 the ssh/startup overhead makes local CPU
faster; above it remote GPU wins, and the margin grows with N (2.4×/4.1× steady at
2 k/20 k). The `remote_screening_min` default of **5,000** sits comfortably above the
crossover, and the campaign's actual screen prewarm (≈10 k–45 k in one batch) is deep
in remote-wins territory.

---

## Result agreement: remote GPU vs local CPU (same checkpoint, same inputs)

Cross-device float32 only — the serialized encode+infer is otherwise bit-identical
(verified by the CPU round-trip test). Measured on the 2,000-candidate pool unless noted.

### Raw ensemble arrays (max abs diff, remote GPU − local CPU)

| array | 2,000-candidate pool | 24 store patterns (determinism test) |
|---|---:|---:|
| `mu_z` (regression means, z-space) | 5.8×10⁻³ | 1.4×10⁻³ |
| `log_sigma` (z-space) | 2.6×10⁻³ | 5.6×10⁻⁴ |
| convergence **probability** (sigmoid) | 6.1×10⁻³ | 4.1×10⁻⁵ |
| convergence raw **logit** (informational²) | 1.05×10⁻¹ | 1.6×10⁻² |

² The raw convergence **logit** differs by ~0.1, but that is a `logit` amplification
of a ≤6×10⁻³ **probability** difference near saturation — the campaign consumes the
convergence head only through `sigmoid` (`predict_convergence`), never the raw logit,
so it is reported but not gated.

### Predicted VALUES through the full `predict()` stack (incl. calibration)

| target | max \|local − remote\| | mean \|local − remote\| | decision tolerance | verdict |
|---|---:|---:|---|---|
| **f_r**    | 1.4×10⁻³        | 1.2×10⁻⁴        | f_r_limit 1.55       | negligible |
| **cyclen** | 1.4×10⁻² EFPD   | 5.1×10⁻³ EFPD   | cycle_tolerance 2.0 EFPD | negligible |

Example rows (predicted f_r / cyclen, local vs remote):

| pair | f_r local | f_r remote | cyclen local | cyclen remote |
|---|---:|---:|---:|---:|
| G3_G4 | 2.18738 | 2.18756 | 669.766 | 669.773 |
| G3_G4 | 2.31473 | 2.31464 | 667.067 | 667.074 |
| H1_H2 | 3.40457 | 3.40422 | 669.720 | 669.723 |
| H1_H2 | 3.47669 | 3.47624 | 664.526 | 664.529 |

The largest decision-relevant disagreement (cyclen, 0.014 EFPD) is **140× below** the
2.0-EFPD cycle-length tolerance and f_r (0.0014) is negligible against the 1.55 limit —
remote GPU and local CPU rank and gate candidates identically. (The all-7-target max
carries a `NaN` from the structurally-censored `max_pin_burnup` / auxiliary column set
by the OOD guard in `predict()`, not from any backend disagreement — the raw `mu_z` has
no NaN and the pin column is gate-advisory only.)

---

## Determinism test

`tests/test_remote_infer.py::test_remote_gpu_matches_local_cpu_determinism` builds the
local-CPU backend and the remote-GPU result from the **same** champion dir on the
**same** 24 inputs and asserts:

* `mu_z`, `log_sigma` agree to < 5×10⁻³ (measured 1.4×10⁻³ / 5.6×10⁻⁴),
* convergence **probability** agrees to < 1×10⁻³ (measured 4.1×10⁻⁵).

It pins GPU 1 via the deck (`RemoteSettings.from_input(lpopt.inp)`), skips cleanly when
the server does not answer a 5 s probe, and prints every measured diff. **It ran for
real against GPU 1 and passed.**

## Reproduce

```
# determinism (real remote GPU, skips if unreachable)
python -m pytest tests/test_remote_infer.py::test_remote_gpu_matches_local_cpu_determinism -s
```

The benchmark harness + remote decomposition probe live in the session scratchpad
(`bench_case.py`, `_infer_probe.py`); the probe is staged to `~/lpopt_infer_probe/` on
the server (a SEPARATE directory — `~/lpopt_ws/lpopt` was never touched) and imports the
already-installed `lpopt` read-only.

## Test counts

`780` baseline → **`787 passed`** (`+6` config tests for the `inference` key, `+1`
real-remote determinism test), `1 skipped` (opt-in `LPOPT_LIVE_SMOKE`), `2 failed`
(the pre-existing `test_boundary_probe` cases that reference live curriculum state —
untouched, as instructed).
