@echo off
REM 1-move ablation wave on BOX 199 -- T6_T4/f121/paramA, 150 verified single
REM moves off 10 diverse elites, stratified by (move_class x radial direction).
REM Pre-registration: data/reports/ablation_wave_prereg_20260815.md
REM
REM This box was freed by the T6_T4 r8 campaign close-out (S1E addendum §8);
REM that campaign is DONE and must not be resumed.  Nothing here touches
REM runs/fpcamp_minfr_T6T4*.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u ablation_wave.py run ^
  --plan data/design/ablation_wave_20260815.json ^
  --package data/design/package ^
  --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/ablation_1move_T6T4 ^
  --workers 16 --host-reserve 1 --wave-size 16 --max-cycles 16 ^
  > ablation_1move_out.log 2>&1
echo %ERRORLEVEL% > ablation_1move_rc.txt
endlocal
