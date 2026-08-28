@echo off
REM MEASURED PIN-BURNUP WAVE -- 44 stored cores re-evaluated with
REM enable_pin_burnup=True, on BOX 199.
REM
REM This is NOT `lpopt optimize`.  It replays each core's OWN stored pattern
REM through curriculum.make_pin_burnup_verifier -- the only reachable path to the
REM %EDT_OPT ipin=1 MAS_PPI edit (verify.py:851 hard-codes it off on the optimize
REM path, which is why every fpcamp row's max_pin_burnup is null).
REM
REM Pre-registration (written and saved BEFORE this ran):
REM   data/reports/pinbu_wave_prereg_20260820.md   (rules, hypotheses, verdicts)
REM   data/reports/pinbu_wave_prereg_20260820.json (44 targets, pinned s1i predictions)
REM
REM The plan JSON is the ONLY source of targets: no search, no sampling, no model
REM is loaded on this path.  Resumable -- a core with a results row is not re-spent.
REM
REM Nothing here touches runs/fpcamp_*, runs/batchswap*, runs/ablation*, or boxes
REM 198 / 181 / 238.  The canonical store is READ-ONLY here (fuel_types.parquet
REM only); merging is a separate, backed-up `pinbu_wave.py patch` step.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u pinbu_wave.py run ^
  --plan data/reports/pinbu_wave_prereg_20260820.json ^
  --deck pinbu_wave_199.inp ^
  --run-dir runs/pinbu_wave ^
  > pinbu_wave_out.log 2>&1
echo %ERRORLEVEL% > pinbu_wave_rc.txt
endlocal
