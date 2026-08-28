@echo off
REM LOW-FEED OPENING CAMPAIGN -- N1_N2 / feed 113 / ga80 on BOX 199.
REM
REM Objective: PURE F_r minimisation under safety gates only (F_r<=1.55,
REM CBC<=1600, F_q<=2.41, |AO|<=0.30).  NO cycle-length target and NO band --
REM minfr_lambda = 1000 (the code's "F_r strictly dominates" default; lambda=0
REM would silently invert the objective into cycle-length maximisation).
REM
REM Pre-registration lives in the deck header itself
REM (fpcamp_minfr_N1N2_f113_199.inp) and in
REM data/reports/fpcamp_N1N2_f113_results_20260816.md after the run.
REM
REM Preconditions shipped 2026-08-16 before arming:
REM   * data/models/s1f            (was absent from this kit; 208,006,780 B)
REM   * data/store/records.parquet (refreshed 21,942,421 B; kit copy was stale
REM                                 and missing all 144 feedgrid f113 rows.
REM                                 Backup: records.parquet.bak_pre_N1N2f113_20260816)
REM
REM Nothing here touches runs/fpcamp_minfr_T6T4* (that campaign is CLOSED),
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
  --input fpcamp_minfr_N1N2_f113_199.inp ^
  --run-dir runs/fpcamp_minfr_N1N2_f113 ^
  --no-early-stop ^
  > fpcamp_minfr_N1N2_f113_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_N1N2_f113_rc.txt
endlocal
