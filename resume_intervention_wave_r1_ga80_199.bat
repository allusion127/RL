@echo off
setlocal
REM resume of the ga80 kit: the original run bat omitted --allow-fallback for ga80, so
REM E1E2_f109 (registered restart pair_feed:MAS_RST.APRQ_11_0615.88 == its parents) refused.
REM check_restart still enforces run restart == parent restart (prereg sec.5 gate).
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
set PLAN=data/design/intervention_wave_r1.json
set RUNDIR=runs/intervention_wave_r1
echo ==== kit ga80 RESUME (E1E2_f109, N1N2_f113) --allow-fallback ==== >> intervention_wave_r1_out.log
"%PY%" -u intervention_wave.py run --plan %PLAN% --kit ga80 --cells E1E2_f109,N1N2_f113 --package FEASIBLE_PACKAGE --fuel-types data/store/fuel_types.parquet --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe --run-dir %RUNDIR% --workers 24 --host-reserve 1 --wave-size 24 --max-cycles 16 --allow-fallback >> intervention_wave_r1_out.log 2>&1
set RC_G=%ERRORLEVEL%
echo paramA=0 ga80_resume=%RC_G% > intervention_wave_r1_rc.txt
endlocal
