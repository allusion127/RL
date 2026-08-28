# N1_N2/f113 5-core measured pin re-verification — PRE-REGISTRATION (2026-08-20)

Registered before any MASTER call. Machine-readable twin with every pinned
number: `data/reports/pinbu_wave_f113pin5_prereg_20260820.json`
(`schema: pinbu_wave_prereg_v1`, 5 targets, pinned against
`data/store/records.parquet` @ 22,216,211 bytes / 74,657 rows).

## 1. What this closes

`data/reports/f113_pin_results_20260820.md` §5 registered — but did NOT run —
a 5-core measured-pin spot-check of the `fpcamp_minfr_N1N2_f113_pin` campaign's
34 four-gate-feasible rows, ranked by lowest validated s1i predicted pin. This
wave executes that registered recommendation, unmodified, using the EXISTING
`pinbu_wave.py` harness (same protocol as the 44-chain
`pinbu_wave_prereg_20260820` / `pinbu_wave_results_20260820` precedent: fixed
pattern replay via `curriculum.make_pin_burnup_verifier`, one chain per core,
determinism as the control).

## 2. Targets — predictions CITED, not recomputed

Per task scope, predicted pin values are cited verbatim from
`f113_pin_results_20260820.md` §5 (already computed there against the
validated static s1i checkpoint) rather than re-run through the model. Stored
F_r was cross-checked against the store row for each record and matches to
4 dp (citation sanity check, done before any MASTER call).

| rank | record_id (short) | stored F_r | predicted pin (cited) |
|---:|---|---:|---:|
| 1 | `2ad9de110b1d` | 1.5074 | **77.09** (UCB z=0.25: 78.89) |
| 2 | `6de15f03c5b6` | 1.5440 | 78.06 |
| 3 | `5c077310d891` | 1.5432 | 78.16 |
| 4 | `817f32c7de0c` | 1.5320 | 78.25 |
| 5 | `e36f10d2b3ad` | 1.5392 | 78.26 |

> **E21 (2026-08-20)**: 이 인용값들은 재현 스크립트가 체크인되어 있지 않은
> 수작업 stub 이었으나, `pinbu_audit_20260820.md` §4.5 가 **정적 `data/models/s1i` 로 5개 전부
> 독립 재산출**해 3자리까지 일치를 확인했다 (77.087 / 78.063 / 78.158 / 78.250 /
> 78.264). **값은 건전하고, 약한 것은 출처 사슬뿐이다.**

All 5: campaign `fpcamp_minfr_N1N2_f113_pin`, `N1_N2`/feed 113, library `ga80`,
restart `pair_feed:MAS_RST.APRQ_11_0677.23`, `max_pin_burnup` currently null in
the store. Identity gate (`compute_record_id` reproduces the planned id from
each row's own stored pattern) verified locally via `pinbu_wave.py run
--dry-run` before launch — no drift on any of the 5.

## 3. Decision rules (fixed now, before the data)

1. **Determinism gate** (unchanged from the 44-chain precedent): |Δf_r| ≤
   0.002, |Δcyclen| ≤ 0.5 EFPD, |Δcbc_max| ≤ 2.0 ppm, plus restart-provenance
   match. A chain failing either gets `determinism_ok=False` /
   `provenance_ok=False` and is refused at merge — no verdict, no patch.
2. **Delivery verdict, per core:** measured `max_pin_burnup` ≤ 80.0 (LEU+
   limit) → PASS; > 80.0 → FAIL.
3. **Deliverable gate for `2ad9de110b1d`** (the task's headline question):
   measured pin ≤ 80.0 AND F_r ≤ 1.55 (already true, 1.5074) →
   first measured-deliverable low-feed core for this programme. Measured pin
   ≤ 78.0 → clears even the acquisition's own model-margin gate, not just the
   licensing limit.
4. No other axis (F_r/CBC/cyclen/F_q/AO) is upgraded by this wave; their
   re-measurement is a determinism control only.

## 4. Spend

5 chains, box **199** (idle at registration: `master=0 python=0`; 57.7 GB free
RAM, 42.7 GB free disk). Run dir `runs/pinbu_wave_f113pin5` (separate from the
44-chain precedent's `runs/pinbu_wave`, so neither run's resume ledger can mix
with the other's). Canonical store read-only until the separate
`pinbu_wave.py patch` step (backs up first, patches `max_pin_burnup` in place
by `record_id`).
