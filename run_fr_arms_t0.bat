@echo off
REM TIER 0 of the flat-assembly -> core F_r experiment (plan flat_assembly_fr_plan_20260802.md).
REM One loading pattern (flattest feasible core b0ff11ef16de, E1_E2/f121) x three fuel sets:
REM   A0 E1_E2 control (must reproduce F_r 1.5207 / node_peak 1.2085 or the experiment is void)
REM   A1 E3_E4 flatter  (FF 1.101/1.139) -> predicted F_r 1.5036
REM   A2 A8_A2 peakier  (FF 1.157/1.178) -> predicted F_r 1.5551  (negative control)
REM
REM chcp 65001 + PYTHONIOENCODING: a cp949 console raises UnicodeEncodeError inside the
REM verifier's log path (observed 2026-08-01, exit 90).
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_arms.py --arm A0 --arm A1 --arm A2 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_arms_t0 --workers 12 --max-cycles 16 > runs_fr_arms_t0.log 2>&1
echo %ERRORLEVEL% > runs_fr_arms_t0_rc.txt
endlocal
