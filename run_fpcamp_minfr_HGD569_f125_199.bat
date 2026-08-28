@echo off
REM HIGH-Gd BORON-OPENED CELL, F_r ASSAULT, PIVOT TO FEED 125 --
REM P6253Z1G06N24_P6253Z2G10N24 / feed 125 / paramA (e_core 5.694, n_gd 24) on
REM BOX 199.  Sibling of run_fpcamp_minfr_HGD569_f109_199.bat (closed,
REM F_r 2.0481 -> 1.6743).
REM
REM Objective: minimise F_r under the four safety gates (F_r<=1.55, CBC<=1600,
REM F_q<=2.41, |AO|<=0.30) PLUS the predicted pin-BU gate (<=78 GWd/tU).
REM NO cycle-length target and NO band -- minfr_lambda = 1000.
REM
REM Budget 60 (7 waves x 8 + 4 reserve), SAME as f109.
REM
REM Why f125 given f109 already ran: the residence-time argument
REM (241/feed: 2.211 cycles @ f109 -> 1.928 @ f125, -12.8%) predicts LOWER
REM pin burnup at the same F_r -- f109's winner sat at predicted pin 81.13,
REM the closest either min_fr precedent has come to the 78 gate.  f125 is the
REM deliverability test, not just another F_r attempt.
REM   data/reports/fpcamp_HGD569_f125_prereg_20260817.md
REM
REM Pre-registration: data/reports/fpcamp_HGD569_f125_prereg_20260817.md
REM plus the deck header (fpcamp_minfr_HGD569_f125_199.inp).
REM
REM Store is the SAME 74,357-row copy shipped for f109 (NOT refreshed with
REM f109's own +60 rows) -- disclosed limitation, see deck ELITE SEEDING block.
REM
REM Nothing here touches runs/fpcamp_minfr_hgd569_f109 (CLOSED and merged),
REM runs/fpcamp_minfr_E1E2_f109 (CLOSED), runs/fpcamp_minfr_N1N2_f113 (CLOSED),
REM or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfr_HGD569_f125_199.inp ^
  --run-dir runs/fpcamp_minfr_hgd569_f125 ^
  --no-early-stop ^
  > fpcamp_minfr_hgd569_f125_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_hgd569_f125_rc.txt
endlocal
