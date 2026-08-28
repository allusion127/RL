@echo off
setlocal
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
"C:\Users\USER\lpopt_work\kit_frontier\venv\Scripts\python.exe" -u -m lpopt produce --input mapcov_181.inp > mapcov_181_out.log 2>&1
echo %ERRORLEVEL% > mapcov_181_rc.txt
endlocal
