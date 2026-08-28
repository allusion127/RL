@echo off
REM Fuel-lever TRANSFER sweep: the top-24 stored E1_E2/f121 patterns re-evaluated
REM under the flatter E3_E4 lattice pair (FF 1.101/1.139 vs the control 1.146/1.152).
REM
REM fr_arms tier 0 measured the swap on ONE pattern (F_r 1.5207 -> 1.5137,
REM map_cov 0.22208 -> 0.19668, but node_peak 1.2085 -> 1.2604 because that pattern
REM was optimized for E1_E2).  This run answers whether the flat fuel wins across the
REM whole elite set: 24 patterns, half picked flattest (node_peak asc), half picked
REM lowest-F_r, each its own paired control.  Only the two fresh batch identities
REM change; the 69 shuffle cards, feed 121, package and protocol are identical.
REM
REM E3_E4 resolves at fallback_level 0 (native restart) in FEASIBLE_PACKAGE -- verified.
REM
REM chcp 65001 + PYTHONIOENCODING: a cp949 console raises UnicodeEncodeError inside the
REM verifier's log path (observed 2026-08-01, exit 90).
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_transfer.py --target-pair E3_E4 --select mixed --k 24 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_transfer_E3E4 --workers 16 --max-cycles 16 > runs_fr_transfer_E3E4.log 2>&1
echo %ERRORLEVEL% > runs_fr_transfer_E3E4_rc.txt
endlocal
