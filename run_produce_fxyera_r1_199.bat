@echo off
REM F_xy-ERA COVERAGE-FILL PRODUCE ROUND 1 -- paramA, box 199.
REM
REM Objective NEUTRAL (DoE coverage fill, standing directive 2026-07-25).  There
REM is no acquisition, no model inference and no objective function on this path.
REM
REM THE DELIVERABLE IS THE RETAINED MAS_OUT SET.  F_xy (MASTER FXYP, hard limit
REM 1.65) is printed only in MAS_OUT; MAS_SUM has no FXYP column, so the store's
REM 74,657 rows carry zero f_xy labels.  [verify] harvest_maps = true forces
REM keep_success (verify.py:754), and [master] keep_success = true pins it, so
REM every converged chain's FINAL equilibrium cycle dir survives with its
REM MAS_OUT for the F_xy parser to retro-label.  A chain produced without that
REM retention loses its F_xy permanently.
REM
REM 19 strata x n_target 40 = 760 converged labels.
REM   A space-filling 5 / B boundary 5 / C exploit 4 / D rule-seed 3 / E OOD 2
REM   = 26.3 / 26.3 / 21.1 / 15.8 / 10.5 % (review 7.6: 25/25/20/15/10).
REM Budget: 760 / 0.678 converged-per-chain ~ 1,120 chains; at the MEASURED 199
REM cadence (24 workers, 24.32 min/wave median = 55.6 chains/h sustained) that
REM is ~20.1 h.  --max-chains 1250 is the HARD budget fence: it stops the round
REM even if convergence comes in below the 0.678 the budget assumed.
REM
REM `lpopt produce` has NO --run-dir flag; [flow] output_root in the deck is
REM "runs/produce_fxyera_r1", so the driver creates
REM runs/produce_fxyera_r1/produce_run_<stamp>/.  Resume is via the ledger + the
REM store, never via the run dir -- the launcher deletes runs/produce_fxyera_r1
REM before every arming so a re-arm is always a clean round.
REM
REM   Pre-registration: data/reports/produce_fxyera_r1_prereg_20260829.md
REM   Deck sha256 gate: launch_produce_fxyera_r1_199.ps1
REM
REM Nothing here touches runs/fpcamp_minfr_triple_f125_r2, any other runs/ dir,
REM or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt produce ^
  --input produce_fxyera_r1_199.inp ^
  --max-chains 1250 ^
  > produce_fxyera_r1_out.log 2>&1
echo %ERRORLEVEL% > produce_fxyera_r1_rc.txt
endlocal
