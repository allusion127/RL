@echo off
REM PHASE-2 F_xy-ERA r1 WAVE -- 40 stored r1 cores re-evaluated with
REM enable_pin_burnup=True and keep_success=true, on BOX 199.
REM
REM This is NOT `lpopt optimize` and NOT `lpopt produce`.  It replays each core's
REM OWN stored pattern through curriculum.make_pin_burnup_verifier -- the only
REM reachable path to the %EDT_OPT ipin=1 MAS_PPI edit (verify.py:851 hard-codes
REM it off on the optimize/produce path, which is why every fxy_* row's
REM max_pin_burnup is null).
REM
REM Registered in:
REM   data/reports/produce_fxyera_r1_prereg_20260829.md      section 6 (phase-2 plan)
REM   data/reports/pinbu_wave_fxyera_r1_prereg_20260830.md   (rules, marks, hashes)
REM   data/reports/pinbu_wave_fxyera_r1_manifest.json        (40 targets, pinned s1i predictions)
REM
REM The manifest is the ONLY source of targets: no search, no sampling, no model
REM is loaded on this path.  Resumable -- a core with a results row is not re-spent.
REM
REM `runs` is RELATIVE on purpose: if runs\ is a junction to another volume the
REM wave follows it with no deck or launcher change.
REM
REM Nothing here touches runs/produce_fxyera_r1, runs/fpcamp_*, runs/pinbu_wave,
REM or boxes 198 / 181 / 238.  The canonical store is READ-ONLY here
REM (fuel_types.parquet only); merging is a separate, backed-up
REM `pinbu_wave.py patch --dry-run` -> `patch` step.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u pinbu_wave.py run ^
  --plan data/reports/pinbu_wave_fxyera_r1_manifest.json ^
  --deck pinbu_wave_fxyera_r1_199.inp ^
  --run-dir runs/pinbu_wave_fxyera_r1 ^
  > pinbu_wave_fxyera_r1_out.log 2>&1
echo %ERRORLEVEL% > pinbu_wave_fxyera_r1_rc.txt
endlocal
