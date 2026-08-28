@echo off
REM 181 coverage-fill (paramA 5.75-6 maps + NEW f121 cells). No queue wait: 181 has
REM no lpopt round running -- but launch ONLY after its SCALE job clears.
setlocal
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
"C:\Users\USER\lpopt_work\kit_frontier\venv\Scripts\python.exe" -u -m lpopt produce --input fill_181.inp > fill_181_out.log 2>&1
echo %ERRORLEVEL% > fill_181_rc.txt
endlocal
