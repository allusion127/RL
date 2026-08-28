@echo off
REM batch_swap deep-sample wave on BOX 199 -- T6_T4/f121/paramA, 220 chains over
REM 6 frontier parents.  Pre-reg: data/reports/batchswap_wave_prereg_20260815.md
REM Does NOT touch runs/fpcamp_minfr_T6T4* or runs/ablation_1move_T6T4.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u batchswap_wave.py run ^
  --plan data/design/batchswap_wave_20260815.json ^
  --package data/design/package ^
  --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/batchswap_enum_T6T4 ^
  --workers 16 --host-reserve 1 --wave-size 16 --max-cycles 16 ^
  > batchswap_out.log 2>&1
echo %ERRORLEVEL% > batchswap_rc.txt
endlocal
