@echo off
REM QUEUE: wait for the running flat_power round to finish, then start coverage-fill.
REM fp_198_rc.txt is written by run_fp_198.bat on completion -> that is the handoff.
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
echo [queue] waiting for flat_power round to finish (fp_198_rc.txt) > fill_198_queue.log
:WAIT
if exist fp_198_rc.txt goto GO
timeout /t 120 /nobreak >nul
goto WAIT
:GO
echo [queue] flat_power done, starting coverage-fill %DATE% %TIME% >> fill_198_queue.log
"C:\Users\USER\lpopt_work\kit_frontier\venv\Scripts\python.exe" -u -m lpopt produce --input fill_198.inp > fill_198_out.log 2>&1
echo %ERRORLEVEL% > fill_198_rc.txt
endlocal
