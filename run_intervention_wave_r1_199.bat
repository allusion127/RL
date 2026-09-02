@echo off
REM CAMPAIGN A ROUND 1 -- Causal Move Atlas, box 199.
REM
REM 800 paid MASTER equilibrium chains: 5 cells x 20 parents x 8 balanced,
REM direction-PAIRED single moves.  Objective NEUTRAL -- this is a measurement
REM wave, not a search; there is no acquisition and no model inference on this
REM path (the blind s1i predictions were taken on the coordinator BEFORE launch
REM and their sha256 is the pre-commitment).
REM
REM THE DELIVERABLE IS PAIRED LABELS *WITH* F_xy.  F_xy (MASTER FXYP, hard limit
REM 1.65) is printed only in MAS_OUT; MAS_SUM has no FXYP column.  The runner is
REM built with harvest_maps=True, which forces keep_success, so every converged
REM chain's FINAL equilibrium cycle dir survives with its MAS_OUT; each cell's
REM fxy_sidecar.jsonl is written straight off lpopt.data.fxy's parse of it.  A
REM chain produced without that retention loses its F_xy permanently.
REM
REM TWO KITS, RUN SEQUENTIALLY.  Asset routing is per library, and the runner
REM builds ONE resolver from ONE --package, so the paramA cells and the ga80
REM cells are two invocations.  `run` refuses a mixed selection rather than
REM staging one library's boards against the other's deck, and it refuses a
REM --package whose library dims do not match the kit it was asked for.
REM
REM   kit paramA : T6T4_f121, HGD569_f125      -> data/design/package  (40,42)
REM   kit ga80   : E1E2_f121, E1E2_f109, N1N2_f113 -> FEASIBLE_PACKAGE (83,85)
REM
REM --allow-fallback is REQUIRED and is not a relaxation: HGD569 f125 has no
REM native restart and its 120 store parents were themselves labelled at
REM fallback_level 3 on pair_ecore:MAS_RST.APRQ_11_0705.02.  The real guard is
REM intervention_wave's own restart-provenance gate, which refuses any cell whose
REM run restart differs from the one its parents carry -- a delta measured across
REM a restart change is a move plus a burnt-fuel history change.
REM
REM Budget: 800 chains at the MEASURED 199 cadence (24 workers, 24.32 min/wave
REM median = 55.6 chains/h sustained) is ~14.4 h.  No --max-chains fence is set:
REM the plan itself IS the fence (800 enumerated record_ids, per-cell cap 200).
REM
REM RESUME: run THIS FILE again, never the launcher -- the launcher deletes
REM runs\intervention_wave_r1 first.  Each cell resumes from its own
REM ablation_results.jsonl, and ablation_wave's _done() re-runs harness failures
REM (disk full / exit status / staging) while keeping physics answers settled.
REM
REM   Pre-registration: data/reports/intervention_wave_r1_prereg_20260829.md
REM   Plan + code sha256 gates: launch_intervention_wave_r1_199.ps1
REM
REM Nothing here touches runs\produce_fxyera_r1, runs\ablation_1move_T6T4,
REM runs\batchswap_enum_T6T4, any other runs\ dir, or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
set PLAN=data/design/intervention_wave_r1.json
set RUNDIR=runs/intervention_wave_r1

echo ==== kit paramA (T6T4_f121, HGD569_f125) ==== >> intervention_wave_r1_out.log
"%PY%" -u intervention_wave.py run ^
  --plan %PLAN% ^
  --kit paramA ^
  --package data/design/package ^
  --fuel-types data/store/fuel_types.parquet ^
  --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir %RUNDIR% ^
  --workers 24 --host-reserve 1 --wave-size 24 --max-cycles 16 ^
  --allow-fallback ^
  >> intervention_wave_r1_out.log 2>&1
set RC_A=%ERRORLEVEL%

echo ==== kit ga80 (E1E2_f121, E1E2_f109, N1N2_f113) ==== >> intervention_wave_r1_out.log
"%PY%" -u intervention_wave.py run ^
  --plan %PLAN% ^
  --kit ga80 ^
  --package FEASIBLE_PACKAGE ^
  --fuel-types data/store/fuel_types.parquet ^
  --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir %RUNDIR% ^
  --workers 24 --host-reserve 1 --wave-size 24 --max-cycles 16 ^
  >> intervention_wave_r1_out.log 2>&1
set RC_G=%ERRORLEVEL%

echo paramA=%RC_A% ga80=%RC_G% > intervention_wave_r1_rc.txt
endlocal
