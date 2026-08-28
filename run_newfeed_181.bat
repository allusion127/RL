@echo off
REM 181 NEW-FEED: ga80 feeds 105/129/137 x 3 bands = 9 cells (emptiest grid columns).
REM Map-coverage driven (F_r is off the objective as of 2026-07-25).
setlocal
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
"C:\Users\USER\lpopt_work\kit_frontier\venv\Scripts\python.exe" -u -m lpopt produce --input newfeed_181.inp > newfeed_181_out.log 2>&1
echo %ERRORLEVEL% > newfeed_181_rc.txt
endlocal
