@echo off
REM PIN-GATED RE-CAMPAIGN -- N1_N2 / feed 113 / ga80 on BOX 199.
REM The 2026-08-16 campaign at this cell (runs/fpcamp_minfr_N1N2_f113, CLOSED)
REM headlined F_r 1.4961 but min_fr_max_cycle gated NOTHING on pin burnup at
REM that time. The 2026-08-20 measured pin wave confirmed that winner FAILS
REM the LEU+ 80 GWd/tU limit (measured 86.189). This run adds the 5th gate,
REM minfr_pin_bu_limit = 78.0, now current champion s1i (v8), fresh seed 1202.
REM
REM Objective: minimise F_r under FIVE gates (F_r<=1.55, CBC<=1600, F_q<=2.41,
REM |AO|<=0.30, predicted pin-BU<=78). NO cycle-length target and NO band --
REM minfr_lambda = 1000.
REM
REM Budget 60 (7 waves x 8 + 4 reserve).
REM Model: data/models/s1i -- the CURRENT champion (v8), the same checkpoint
REM the pin-head validation wave (pinbu_wave_results_20260820.md) scored.
REM   Pre-registration: data/reports/f113_pin_prereg_20260820.md
REM
REM Nothing here touches runs/fpcamp_minfr_N1N2_f113 (CLOSED and merged),
REM runs/fpcamp_minfr_hgd569_f125_seedctl (CLOSED), runs/fpcamp_minfr_triple_f125
REM (CLOSED), runs/pinbu_wave (CLOSED), or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfr_N1N2_f113_PIN_199.inp ^
  --run-dir runs/fpcamp_minfr_N1N2_f113_pin ^
  --no-early-stop ^
  > fpcamp_minfr_N1N2_f113_pin_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_N1N2_f113_pin_rc.txt
endlocal
