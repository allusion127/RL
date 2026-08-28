@echo off
REM TIER 0 step T0-3: A0 replicate from the same restart into a different run dir.
REM This measures the harness reproducibility floor.  Falsifier F3 fires if the
REM A0-vs-replicate spread exceeds 0.005 in F_r -- in that case the reactivity-matched
REM Tier-1 arm (predicted -0.010) is inside the noise and can never be decisive.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_arms.py --arm A0 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_arms_t0_rep --workers 8 --max-cycles 16 > runs_fr_arms_t0rep.log 2>&1
echo %ERRORLEVEL% > runs_fr_arms_t0rep_rc.txt
endlocal
