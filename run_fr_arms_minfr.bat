@echo off
REM The same three-point fuel dose test, but anchored on the store's LOWEST-F_r
REM feasible core (deb058c00433: F_r 1.4636, node_peak 1.2620, cyclen 633.33,
REM CBC 1326.7) instead of the flattest one.
REM
REM Why this second anchor matters: inside the LP space nodal flatness and F_r
REM trade off -- the flattest core (node_peak 1.2085) sits at F_r 1.5207 while
REM the best F_r core is 0.057 lower at a WORSE node_peak.  The fuel (FF) lever
REM is orthogonal to the pattern lever, so the question that actually answers
REM "does flat-assembly sampling give a lower-F_r core" is whether FF still buys
REM its ~0.9 x dFF at the F_r optimum, not only at the flatness optimum.
REM A1 landing below 1.4636 would be the lowest core F_r ever measured here.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_arms.py --reference minfr --arm A0 --arm A1 --arm A2 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_arms_minfr --workers 12 --max-cycles 16 > runs_fr_arms_minfr.log 2>&1
echo %ERRORLEVEL% > runs_fr_arms_minfr_rc.txt
endlocal
