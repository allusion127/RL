@echo off
REM QUEUE stage 2 on 198: wait for the coverage-fill to finish (fill_198_rc.txt),
REM then run the paramA NEW-FEED campaign (f105/f113/f129/f137, 12 cells).
setlocal
REM Console/stdout encoding (incident 2026-08-30): stdout here is REDIRECTED to a
REM log file, so Windows picks the ANSI codepage (cp949 on these boxes) for it and
REM a single em-dash in an lpopt log line raised UnicodeEncodeError -- that killed
REM a FINISHED 100-call campaign before report.md / delivery.json were written.
REM PYTHONIOENCODING pins the interpreter's stdio to utf-8; chcp 65001 makes the
REM console agree when the run is watched live.  Both are setlocal-scoped and do
REM not reach MASTER.
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
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
