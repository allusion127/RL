@echo off
REM R-SEED CONTROL -- 2-TYPE, SAME DONOR-ENRICHED ELITE POOL THE 3-TYPE RUN GOT
REM P6253Z1G06N24_P6253Z2G10N24 / feed 125 / paramA on BOX 199.
REM Demanded by tripletype_f125_results_20260817.md section 8 (R-SEED): the
REM 3-type headline (-0.0364) is confounded with a seeding change; this run
REM changes ONLY the seeding+model, keeping the alphabet at 2, to separate them.
REM
REM Objective: minimise F_r under the four safety gates (F_r<=1.55, CBC<=1600,
REM F_q<=2.41, |AO|<=0.30) PLUS the predicted pin-BU gate (<=78 GWd/tU).
REM NO cycle-length target and NO band -- minfr_lambda = 1000.
REM
REM Budget 60 (7 waves x 8 + 4 reserve), SAME as both predecessors.
REM
REM Model: data/models/s1h -- the CURRENT champion (v7), same checkpoint the
REM 3-type run used.  Seeding: no elite_seed_cases knob (proven no-op for a
REM case naming itself); box 199's store already carries this pair's own
REM 129 converged rows across both feeds, which _case_store_rows picks up
REM natively -- the identical donor population elite_seed_cases fed the
REM 3-type run.
REM   Pre-registration: data/reports/hgd569_f125_seedctl_prereg_20260817.md
REM
REM Nothing here touches runs/fpcamp_minfr_hgd569_f125 (CLOSED and merged),
REM runs/fpcamp_minfr_triple_f125 (CLOSED and merged), runs/fpcamp_minfr_hgd569_f109
REM (CLOSED), runs/fpcamp_minfr_E1E2_f109 (CLOSED), runs/fpcamp_minfr_N1N2_f113
REM (CLOSED), or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfr_HGD569_f125_SEEDCTL_199.inp ^
  --run-dir runs/fpcamp_minfr_hgd569_f125_seedctl ^
  --no-early-stop ^
  > fpcamp_minfr_hgd569_f125_seedctl_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_hgd569_f125_seedctl_rc.txt
endlocal
