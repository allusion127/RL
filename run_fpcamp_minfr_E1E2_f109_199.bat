@echo off
REM LOW-FEED OPENING CAMPAIGN r2 -- E1_E2 / feed 109 / ga80 on BOX 199.
REM
REM Objective: PURE F_r minimisation under safety gates only (F_r<=1.55,
REM CBC<=1600, F_q<=2.41, |AO|<=0.30).  NO cycle-length target and NO band --
REM minfr_lambda = 1000 (lambda=0 would invert the objective into cycle-length
REM maximisation; measured and documented at f113).
REM
REM Pre-registration lives in the deck header (fpcamp_minfr_E1E2_f109_199.inp).
REM Predecessor: fpcamp_minfr_N1N2_f113_199.inp -> 41 feasible, F_r 1.4961,
REM first feasible core at call 23 (data/reports/fpcamp_N1N2_f113_results_20260816.md).
REM
REM Precondition shipped 2026-08-17 before arming:
REM   * data/models/s1g  (was absent from this kit; 207,997,335 B, 30 files)
REM
REM Nothing here touches runs/fpcamp_minfr_N1N2_f113 (CLOSED and merged),
REM runs/batchswap_enum_625_T6T4, or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfr_E1E2_f109_199.inp ^
  --run-dir runs/fpcamp_minfr_E1E2_f109 ^
  --no-early-stop ^
  > fpcamp_minfr_E1E2_f109_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_E1E2_f109_rc.txt
endlocal
