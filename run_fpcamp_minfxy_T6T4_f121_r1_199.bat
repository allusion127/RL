@echo off
REM min_fxy WAVE CAMPAIGN, ROUND 1 -- T6_T4 / feed 121 / paramA on BOX 199.
REM THE PROGRAMME'S FIRST min_fxy CAMPAIGN.
REM
REM Objective: minimise F_xy (MASTER FXYP, pin PLANAR peaking) under the hard
REM limit F_xy <= 1.65, with F_r <= 1.55 KEPT as a constraint plus CBC <= 1600,
REM F_q <= 2.41, |AO| <= 0.30 and the PREDICTED pin-BU gate (<= 78 GWd/tU).
REM NO cycle band gates anything -- minfxy_lambda = 1000, cyclen is the
REM secondary tie-break only.
REM
REM Budget 100 (12 waves x 8 + 4 reserve) -- the same 100 every min_fr round
REM at this cell spent (r3..r8), so this round is comparable call-for-call.
REM
REM THE NUMBER TO BEAT: 1.5491 (this cell's incumbent joint-clean F_xy,
REM record_id 46e687ed..., F_r 1.5018, CBC 1315.08, cyclen 621.28).
REM PRIMARY mark: F_xy <= 1.5441 (that minus the registered 0.005 margin).
REM STRETCH: 1.5295, the programme-wide best, which lives at E1_E2/f121.
REM
REM harvest_maps = true is LOAD-BEARING here: F_xy is absent from MAS_SUM and is
REM parsed from the final cycle's MAS_OUT, which survives only because
REM harvest_maps forces the verifier's keep_success.
REM
REM EXPECTED banner on the s1i arm, NOT a fault:
REM   [optimize][F_xy PROXY] the served model exposes NO `predict_fxy` head ...
REM   -> every wave records fxy_source='proxy'.  Read it as an F_r-surrogate
REM      search with F_xy MEASURED after the fact.  On the s1j arm the banner is
REM      absent and fxy_source='head'.
REM
REM NOTE (R) -- `lpopt report`'s "Best verified loading patterns" table ranks by
REM CYCLE DISTANCE regardless of objective and will NOT show the F_xy winner.
REM Read `state.json -> best`.  Known defect, unchanged.
REM
REM NOTE (P) -- no row this round can be `deliverable = true`: the PPI pin edit
REM is not a deck knob (verify.py:851 hard-codes enable_pin_burnup=False), so
REM max_pin_burnup is UNKNOWN on every row.  Phase 2 (`pinbu_wave_keep_199.inp`)
REM supplies it.  Registered in the prereg SS5, not a defect of this run.
REM
REM   Pre-registration: data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md
REM   Design:           data/reports/fxy_switch_design_20260829.md
REM
REM Nothing here touches runs/fpcamp_minfr_triple_f125_r2, runs/produce_fxyera_r1,
REM any other runs/ dir, or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfxy_T6T4_f121_r1_199.inp ^
  --run-dir runs/fpcamp_minfxy_t6t4_f121_r1 ^
  --no-early-stop ^
  > fpcamp_minfxy_t6t4_f121_r1_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfxy_t6t4_f121_r1_rc.txt
endlocal
