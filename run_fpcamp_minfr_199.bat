@echo off
REM ============================================================================
REM min_fr_max_cycle campaign on box 199.  Champion 20260729_054749,
REM E1_E2 / feed 121 / e_core 5.00 (ga80).  100 MASTER calls, 12 waves + reserve.
REM
REM THE POINT: this is the first campaign in the program that SEARCHES for low
REM F_r instead of harvesting it from a flatness run.  Incumbent to beat is
REM F_r 1.4636 @ cyclen 633.33 (record deb058c004333ff4, campaign fpcamp_199).
REM Full justification and the knob audit are in fpcamp_minfr_199.inp's header.
REM
REM ---------------------------------------------------------------------------
REM NO map_calibration STEP.  run_fpcamp2b/3_199.bat documented WHY they skip the
REM refit; this deck does not need the artifact AT ALL.  The ModelMismatchError
REM precondition and the MapHarvestAbort floor are both inside the flat_power
REM branch only (search/campaign.py:511-520 and :1935-1936; the attribute is
REM documented "inert outside flat_power" at campaign.py:385-386).  A
REM min_fr_max_cycle run neither reads map_calibration.json nor can abort on map
REM harvest, so there is no preconditioning step to run or to skip.
REM
REM ---------------------------------------------------------------------------
REM EXPECTED ON STARTUP, NOT A FAILURE — the run opens with:
REM   [optimize][DEPRECATED] objective='min_fr_max_cycle' is a RETIRED production
REM   mode (flatness-first program 10 STOP) ...
REM That banner is emitted by CampaignDriver.__init__ (campaign.py:421-431) for
REM every objective in _RETIRED_PRODUCTION_OBJECTIVES (campaign.py:72).  The same
REM comment block states these modes "still run" and are kept for exactly this
REM use.  The run proceeds normally; do not kill it.
REM
REM chcp 65001 + PYTHONUTF8 + PYTHONIOENCODING ARE MANDATORY HERE, and more so
REM than for the flat campaigns: that DEPRECATED banner itself contains an EM-DASH
REM ("baselines only - use objective='flat_power'", campaign.py:430) and it is
REM printed INSIDE CampaignDriver.__init__.  On a cp949 console print() raises
REM UnicodeEncodeError before wave 0 and the run dies at construction (same class
REM of failure observed 2026-08-01, exit 90).  Without the codepage this deck
REM CANNOT reach its first MASTER call.
REM ============================================================================
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize --input fpcamp_minfr_199.inp --run-dir runs/fpcamp_minfr2_199 --no-early-stop > fpcamp_minfr2_199_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr2_199_rc.txt
endlocal
