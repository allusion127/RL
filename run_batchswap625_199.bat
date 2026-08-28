@echo off
REM batch_swap 625 EFPD-branch sweep on BOX 199 -- T6_T4/f121/paramA, 220 chains
REM over the 5 in-band frontier parents.
REM Pre-reg: data/reports/batchswap625_wave_prereg_20260815.md
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u batchswap625_wave.py run ^
  --plan data/design/batchswap625_wave_20260815.json ^
  --package data/design/package ^
  --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/batchswap_enum_625_T6T4 ^
  --workers 16 --host-reserve 1 --wave-size 16 --max-cycles 16 ^
  > batchswap625_out.log 2>&1
echo %ERRORLEVEL% > batchswap625_rc.txt
endlocal
