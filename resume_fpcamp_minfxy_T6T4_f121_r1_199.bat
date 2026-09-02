@echo off
setlocal
REM Console/stdout encoding (incident 2026-08-30): stdout here is REDIRECTED to a
REM log file, so Windows picks the ANSI codepage (cp949 on these boxes) for it and
REM a single em-dash in an lpopt log line raised UnicodeEncodeError -- that killed
REM a FINISHED 100-call campaign before report.md / delivery.json were written.
REM PYTHONIOENCODING pins the interpreter's stdio to utf-8; chcp 65001 makes the
REM console agree when the run is watched live.  Both are setlocal-scoped and do
REM not reach MASTER.
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize --input fpcamp_minfxy_T6T4_f121_r1_199.inp --run-dir runs/fpcamp_minfxy_t6t4_f121_r1 --no-early-stop --resume >> fpcamp_minfxy_t6t4_f121_r1_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfxy_t6t4_f121_r1_rc.txt
endlocal
