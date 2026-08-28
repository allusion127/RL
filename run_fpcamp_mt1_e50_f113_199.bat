@echo off
REM P3b MULTI-TYPE ANCHOR CAMPAIGN 1/1 -- E3_E1_E2 / feed 113 / ga80
REM on BOX 199.  Cell chosen by the pre-registered anchor rule (PREREG §4.2),
REM executed by anchor_select_multitype.py.
REM
REM Objective: minimise F_r under FIVE gates (F_r<=1.55, CBC<=1600, F_q<=2.41,
REM |AO|<=0.30, predicted pin BU<=78).  NO cycle target, NO band, lambda=1000.
REM Budget 60 (7 waves x 8 + 4 reserve) -- the house budget.
REM
REM THE NUMBER TO BEAT: 1.6233.
REM Model: data/models/s1i (cond_v8).
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_mt1_e50_f113_199.inp ^
  --run-dir runs/fpcamp_mt1_e50_f113 ^
  --no-early-stop ^
  > fpcamp_mt1_e50_f113_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_mt1_e50_f113_rc.txt
endlocal
