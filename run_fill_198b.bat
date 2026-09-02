@echo off
REM 198 STAGE 3: 199's orphaned share (paramA f121 + band 5.75-6).
REM The 139 f121 labels 199 produced are already in the shipped store, so produce
REM resumes and only makes the remainder.
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
"C:\Users\USER\lpopt_work\kit_frontier\venv\Scripts\python.exe" -u -m lpopt produce --input fill_198b.inp > fill_198b_out.log 2>&1
echo %ERRORLEVEL% > fill_198b_rc.txt
endlocal
