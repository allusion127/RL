@echo off
REM min_fr campaign on the T6_T4/f121 operating-point cell (re-screen pick R2).
REM Fuel: T6 (5.30/gd6/n16, FF 1.1011) in the 68 cool slots, T4 (5.00/gd8/n24,
REM FF 1.1430) in the 53 hot slots -- contrast +0.075 per the contrast law.
REM Elite parents: 32 transferred rows (cyclen 620.7 mean, CBC 1331, map_cov
REM improved 32/32) -- F_r starts ~1.59 because the patterns are E1_E2-shaped;
REM re-arranging for THIS fuel is this campaign's whole job.  Fusion-law floor
REM with contrast enforced: ~1.423.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u -m lpopt optimize --input fpcamp_minfr_T6T4.inp --run-dir runs/fpcamp_minfr_T6T4 --no-early-stop > fpcamp_minfr_T6T4_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_T6T4_rc.txt
endlocal
