@echo off
REM 199 coverage-fill (paramA 5.75-6 maps + NEW f121 cells). 181's former share.
REM NON-INTRUSION: confirm cms has not returned (master=0) before running.
setlocal
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
"C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe" -u -m lpopt produce --input fill_199.inp > fill_199_out.log 2>&1
echo %ERRORLEVEL% > fill_199_rc.txt
endlocal
