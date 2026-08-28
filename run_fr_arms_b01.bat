@echo off
REM Plan step T1-7, partial: the two paramA arms that need NO new lattices.
REM Q1_Q2 (FF 1.122/1.174) and Q7_Q8 (FF 1.205/1.209) both resolve at
REM fallback_level 0 with native restarts against the CURRENT paramA package
REM (ncomp 38, unchanged), so they are launchable today.  They extend the
REM dose-response range from FF [1.101,1.178] to [1.101,1.209] and give an
REM independent within-library replication of the transfer slope.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_arms.py --arm B0 --arm B1 ^
  --package data/design/package ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_arms_b01 --workers 8 --max-cycles 16 > runs_fr_arms_b01.log 2>&1
echo %ERRORLEVEL% > runs_fr_arms_b01_rc.txt
endlocal
