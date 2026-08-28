# batch_swap on the 625 EFPD branch — RESULTS

Pre-registration: `data/reports/batchswap625_wave_prereg_20260815.md`.
**This is a negative result and is reported as such.** A mid-run disk-full
incident on box 199 is documented in §5.

---

## 1. RESULT

**220 chains, 218 converged, 2 `non_finite_flux`, 0 harness failures remaining**
(after an 18-chain re-run, §5). All 218 children stayed in band.

| readout | verdict |
|---|---|
| **§4a best in-band F_r** | **1.4741** — beats the in-band marks by a hair (−0.0006 vs 1.4747, −0.0008 vs 1.4749) but **misses the 1.4636 stretch target by +0.0105** |
| **§4c λ-objective** | **FAILS** — net **−1.48 EFPD-eq** vs the r8 record. Even the in-band "improvement" is not usable. |
| **§4b improving fraction** | **0.005** (1 of 218) vs 0.052 on the 618 branch — a 10× collapse, Fisher exact **p = 0.0028** |
| **Registered conclusion** | **The class does not transfer.** The 618-branch result was branch-specific. |

**The program's usable in-band optimum is unchanged: the r8 board `188c9a338d9f`,
F_r 1.4749 @ 625.46 EFPD.** Nothing in this wave displaces it.

---

## 2. §4a — best in-band feasible F_r

| | |
|---|---|
| best in-band | `8ec148f9d9ef…` **F_r 1.4741 @ 623.657 EFPD** |
| feasibility | F_q 1.8179 · CBC 1339.68 · \|AO\| 0.0253 — clears all four axes |
| parent | `1ca37638c03c…` (1.4750) — d_F_r **−0.0009** |

| mark | value | delta | verdict |
|---|---:|---:|---|
| current in-band best | 1.4747 | **−0.0006** | beats |
| r8 campaign record | 1.4749 | **−0.0008** | beats |
| **ga80 incumbent (stretch)** | 1.4636 | **+0.0105** | **misses** |

**Only 1 of 218 boards beat 1.4747. Zero beat 1.4636.** On the 618 branch, the
comparable wave produced 7 boards past the then-record with a best improvement of
−0.0080. Here the single winner improves by −0.0006 — an order of magnitude
smaller, on the same budget, with the same operator.

**Every child stayed in band** (cyclen 623.27–626.19, **0 out-of-band**),
confirming the pre-registered expectation that `batch_swap` does not move cycle
length. That expectation was checked per child, not assumed.

---

## 3. §4c — the usability test, which the result fails

| comparison | d_F_r | worth | d_cyclen | **net** |
|---|---:|---:|---:|---:|
| vs **r8 record** (1.4749 @ 625.459) | −0.0008 | +0.32 EFPD-eq | **−1.80** | **−1.48 EFPD-eq → r8 preferred** |
| vs **ga80 incumbent** (1.4636 @ 633.329) | +0.0105 | −4.20 EFPD-eq | −9.67 | −13.87 EFPD-eq → incumbent preferred |

The 1.4741 board buys 0.0008 of F_r — worth 0.32 EFPD at the deck's λ = 400 —
while giving up 1.80 EFPD of cycle length. **It is a worse board than the r8
record on the campaign's own objective**, despite being nominally an in-band F_r
improvement.

This is the second consecutive wave whose headline F_r number fails the λ test.
Registering §4c as a primary-tier readout was the right call: on F_r alone this
wave would read as "new in-band record", and that framing would be wrong.

---

## 4. §4b — improving fraction, and the branch asymmetry

Quoted for deep-n parents only (n ≥ 40), honouring the n=4 retraction:

| parent | parent F_r | n | min d_F_r | p25 | median | mean | max | frac improving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `abd38bc5b212` | 1.4747 | 70 | +0.0013 | +0.121 | +0.155 | +0.158 | +0.379 | **0.000** |
| `188c9a338d9f` | 1.4749 | 68 | +0.0081 | +0.128 | +0.151 | +0.170 | +0.375 | **0.000** |
| `1ca37638c03c` | 1.4750 | 40 | −0.0009 | +0.123 | +0.165 | +0.168 | +0.409 | 0.025 |

Not quoted as rates (n < 40, carried in the corpus only): `a6d11f78a7f6` n=25,
`195699e488e6` n=15 — both 0.000.

**Pooled: 0.005 ± 0.009 (1 of 218).** The registered expectation was "at or below
0.052"; it is satisfied, by a factor of ten.

**The branch asymmetry is the real finding:**

| | 618 branch | 625 branch |
|---|---:|---:|
| chains | 213 | 218 |
| improving | **11 (0.052)** | **1 (0.005)** |
| best d_F_r | **−0.0265** | −0.0009 |
| boards past the then-record | 7 | 1 |

Fisher exact on the two improving rates: **p = 0.0028**. The difference is not
sampling noise. Two of these parents (`abd38bc5b212`, `188c9a338d9f`) had
**zero** improving children in 138 chains between them.

**Registered reading, third clause, fires:** *"no in-band improvement on 1.4747 →
the class does not transfer; the 618-branch result was branch-specific."* Strictly
there is one improvement, at −0.0006 and λ-negative, which does not change the
reading. **`batch_swap`'s value is a property of the 618 branch, not of the
operator.** The recommendation in `batchswap_wave_results` §9.1 — "a `batch_swap`
pass belongs in the campaign operator mix" — is **weakened accordingly**: it pays
on some branches and not others, and this wave cannot say in advance which.

---

## 5. Incident — disk exhaustion on 199, and what it did not damage

**What happened.** During wave 12, `C:` on 199 reached **0 bytes free**. `runs/`
had accumulated **144 GB** on a ~250 GB disk across many campaigns; my three waves
were ~24 GB of that. The harness retains `master_work` on failure, so once the
disk filled each failing chain retained another work dir and compounded it
(8.7 GB in this run alone).

**Damage assessment — the converged results are sound.**

* 200 converged in the first pass, **all 200 carrying maps**;
* wall-time distributions separate cleanly — converged min 195 s vs failed max
  105 s — so no converged row is a truncated artefact;
* chains 1–192 gave 190/192 (the 2 misses are honest `non_finite_flux`);
* 18 failures, all in chains 193–220: 15 × `[Errno 28] No space left on device`,
  3 × `MASTER exited with status 38`.

**Recovery.** Freed 16.6 GB by deleting **only my own already-harvested and
merged** artefacts (`runs/ablation_1move_T6T4`, `runs/batchswap_enum_T6T4`, and
this run's `master_work`), after verifying the local kit copies and their 150 /
213 merged rows. No other campaign's run dirs were touched. The 18 chains were
re-run: **18/18 converged.**

**A resume bug this exposed, now fixed.** `ablation_wave._done()` treated any
`record_id` present in the results jsonl as done, *including harness failures* —
so a plain re-launch would have silently skipped all 18 and produced a wave that
looked complete but was 18 labels short. For this wave I filtered the 18 by hand
(backup `ablation_results.jsonl.bak_diskfull`, full copy retained locally).

`_done()` now keys off the project's own taxonomy
(`verify.PHYSICS_KILL_FAILURES`): converged, non-converged and physics kills are
settled; staging / disk / exit-status failures are not. Test
`tests/test_ablation_resume.py` — 10 cases, including a replay of the real
incident file (220 rows → 202 settled, 18 to re-run), which reproduces the manual
filtering exactly. `ablation_wave.py` hash changes `89d05e83…` → `1b94c712…`;
recorded as Amendment 2 in `ablation_wave_prereg_20260815.md` §8. **No completed
wave's results depend on the change** — all three ran on the pre-fix code.

**Open risk, not actioned (authorization deferred):** ~120 GB of older campaign
run dirs remain on 199 and are not mine to prune. Disk stands at 16.6 GB free; a
future 220-chain wave needs ~9 GB. Either those dirs get pruned by their owner,
or future waves should run with `keep_success=False` when maps are not required.

---

## 6. Corpus rows added

| | before | after |
|---|---:|---:|
| `data/policy/steps.parquet` | 27,843 | **28,063** (+220) |
| ↳ `batchswap_enum_625` | — | **220** |
| `data/store/records.parquet` | 73,683 | **73,903** (+220) |
| `data/store/maps.npz` | — | +218 |

All 220 are `single_move=True`, `move_class='batch_swap'` (195 inward / 20
outward / 5 neutral), cell `T6_T4/f121/paramA`. Corpus lineage totals now:
`sa_mocha` 21,766 · `lpopt_genome` 5,714 · `batchswap_enum_625` 220 ·
`batchswap_enum` 213 · `ablation_paramA` 150.

The cell now holds **1,653 rows, 500 feasible, 374 in-band**; overall best F_r
**1.4605** (out-of-band, 618 branch), in-band best **1.4741**.

Appender: `ablation_analyze.py corpus --campaign … --lineage …`.
`ablation_wave.py` was edited only for the `_done()` fix (§5), after all waves
ran; `batchswap_wave.py` (`56c11457…`) and `batchswap625_wave.py` (`8d981b79…`)
are unedited.

---

## 7. Limits

Prereg §5 binds. The strongest caveat on the negative result: **coverage is
7–31% per parent**, so this bounds `batch_swap`'s yield on the 625 branch at this
depth — it does not prove the neighbourhoods are barren. But two parents at 70
and 68 chains returning **zero** improvements is a substantially stronger bound
than the 618-branch parents ever needed to produce their 11. **This still does
not reopen the S1E §8 close-out** — different instrument, as registered three
times now.

---

## 8. Artefacts

| path | what |
|---|---|
| `data/reports/batchswap625_wave_prereg_20260815.md` | pre-registration |
| `data/reports/batchswap625_wave_results_20260815.md` | this report |
| `data/reports/batchswap625_wave_tables.txt` | raw readout |
| `batchswap625_wave.py` | plan/run/kit — sha256 `8d981b79…` |
| `batchswap625_analyze.py` | §4a/§4b/§4c readouts |
| `data/design/batchswap625_wave_20260815.json` | plan, sha256 `0cdca15b…` |
| `runs/batchswap_enum_625_T6T4/ablation_results.jsonl` | 220 outcomes, sha256 `1b6738fe…` |
| `runs/…/ablation_results.jsonl.full_with_diskfail` | the pre-recovery file, retained |
| `tests/test_ablation_resume.py` | the `_done()` regression guard |

Post-merge: `records.parquet` `e86e934a…`, `steps.parquet` `6acc0692…`.
Backup `steps.parquet.bak_pre_batchswap_enum_625_T6T4`.
199 free (`master=0`, rc 0). 181 / 198 / 238 untouched.

---

## 9. What this changes

1. **Stop the `batch_swap` line on this cell.** Two waves, 431 chains: the 618
   branch paid, the 625 branch did not (p = 0.0028). A third wave on either is
   not indicated.
2. **The program's in-band answer is still the r8 board** (1.4749 @ 625.46).
   The 1.4605 board remains an F_r record with two caveats — out of band, and
   λ-dominated.
3. **Weaken, don't drop, the operator-mix recommendation.** `batch_swap` is
   worth having available, but its payoff is branch-dependent and cannot be
   predicted from the class alone.
4. **The λ-objective check belongs in every frontier readout.** Two consecutive
   waves produced F_r "records" that fail it.
5. **Disk headroom is now a live operational constraint on 199** (§5).
