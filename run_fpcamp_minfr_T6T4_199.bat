@echo off
REM T6_T4/f121 min_fr campaign round 2 on BOX 199 (user directive 2026-08-11:
REM verdict campaigns run on 199, not the local box).  Round 1 (local) reached
REM F_r 1.5549 / cyclen 618.3 -- 0.005 from the feasibility gate; this round's
REM parents include those 191 fresh labels.  CBC gate 1600, champion split_S1b,
REM paramA routing fix (campaign.py _resolver -> build_case_resolver) REQUIRED:
REM this kit's lpopt must carry the 2026-08-11 patches.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize --input fpcamp_minfr_T6T4_199.inp --run-dir runs/fpcamp_minfr_T6T4 --no-early-stop > fpcamp_minfr_T6T4_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_T6T4_rc.txt
endlocal
