@echo off
REM QUEUE stage 2 on 198: wait for the coverage-fill to finish (fill_198_rc.txt),
REM then run the paramA NEW-FEED campaign (f105/f113/f129/f137, 12 cells).
setlocal
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
:WAIT
if exist fill_198_rc.txt goto GO
timeout /t 180 /nobreak >nul
goto WAIT
:GO
"C:\Users\USER\lpopt_work\kit_frontier\venv\Scripts\python.exe" -u -m lpopt produce --input newfeed_198.inp > newfeed_198_out.log 2>&1
echo %ERRORLEVEL% > newfeed_198_rc.txt
endlocal
