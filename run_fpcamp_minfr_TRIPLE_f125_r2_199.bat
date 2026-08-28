@echo off
REM 3-FRESH-TYPE (GRADED) CAMPAIGN, ROUND 2 -- S3_T1_S5 / feed 125 / paramA
REM on BOX 199.  Successor to run_fpcamp_minfr_TRIPLE_f125_199.bat (round 1,
REM CLOSED: joint-clean F_r 1.5993, CBC 1597.33, 4-of-5 gates clean, measured
REM pin BU 74.38 PASS).
REM
REM Objective: minimise F_r under the four safety gates (F_r<=1.55, CBC<=1600,
REM F_q<=2.41, |AO|<=0.30) PLUS the predicted pin-BU gate (<=78 GWd/tU).
REM NO cycle-length target and NO band -- minfr_lambda = 1000.
REM
REM Budget 60 (7 waves x 8 + 4 reserve), SAME as round 1.  Fresh seed, fresh
REM run dir, champion s1i (v8) -- the first checkpoint trained on any 3-type
REM row -- and an elite pool now seeded with round 1's own 49 converged
REM triples directly, plus the 2-type f125 pair as before.
REM
REM THE NUMBER TO BEAT: 1.5993 (round-1 joint-clean).  TARGET: F_r <= 1.55
REM full-feasible = the programme's first deliverable-grade high-e core.
REM
REM   Pre-registration: data/reports/tripletype_f125_r2_prereg_20260820.md
REM   Round-1 results:  data/reports/tripletype_f125_results_20260817.md
REM
REM Nothing here touches runs/fpcamp_minfr_triple_f125 (round 1, CLOSED and
REM merged), runs/fpcamp_minfr_hgd569_f125 (CLOSED and merged), or boxes
REM 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfr_TRIPLE_f125_r2_199.inp ^
  --run-dir runs/fpcamp_minfr_triple_f125_r2 ^
  --no-early-stop ^
  > fpcamp_minfr_triple_f125_r2_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_triple_f125_r2_rc.txt
endlocal
