@echo off
REM STEP 0 of data/reports/kcurve_fusion_memo_20260809.md -- the H-family NULL TEST.
REM Pre-registered decision rule is in the ARMS dict comment of fr_arms.py, written
REM BEFORE these four chains ran.
REM
REM   D1 H1_H2 -> H2 in all 121 fresh slots   treated
REM   D2 H3_H4 -> H4 in all 121 fresh slots   treated
REM   D3 H1_H2 -> H1 in all 121 fresh slots   negative control
REM   D4 H3_H4 -> H3 in all 121 fresh slots   negative control
REM
REM Stop condition: if the test REFUTES sufficiency, no GPU time goes into the
REM k-curve precision upgrade (memo section 3-1) -- the effort moves to the ADF /
REM pin-form-function channels instead.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_arms.py --arm D1 --arm D2 --arm D3 --arm D4 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_arms_d --workers 12 --max-cycles 16 > runs_fr_arms_d.log 2>&1
echo %ERRORLEVEL% > runs_fr_arms_d_rc.txt
endlocal
