@echo off
REM FIRST 3-FRESH-TYPE (GRADED) CAMPAIGN -- S3 _ <mid> _ S5 / feed 125 / paramA
REM on BOX 199.  Successor to run_fpcamp_minfr_HGD569_f125_199.bat (closed,
REM joint-clean F_r 1.6357, 4-of-5 gates clear, predicted pin 76.96).
REM
REM Objective: minimise F_r under the four safety gates (F_r<=1.55, CBC<=1600,
REM F_q<=2.41, |AO|<=0.30) PLUS the predicted pin-BU gate (<=78 GWd/tU).
REM NO cycle-length target and NO band -- minfr_lambda = 1000.
REM
REM Budget 60 (7 waves x 8 + 4 reserve), SAME as the 2-type predecessor, so the
REM only thing that changes is the size of the fresh-type alphabet: 2 -> 3.
REM
REM THE NUMBER TO BEAT: 1.6357 (the 2-type joint-clean frontier at this exact
REM cell and feed).  NULL = "a third fresh type does not buy F_r here".
REM
REM Model: data/models/s1h -- the v7 champion, the FIRST checkpoint in the
REM programme that can encode a 3-type core (18 globals; s1g's 13 cannot).
REM
REM Elite seeding: [search] elite_seed_cases hands this campaign the donor
REM pair's own converged rows as mutation parents, which graded_morph then
REM re-labels onto the third type.  Without it the pool would be empty.
REM   Pre-registration: data/reports/tripletype_f125_prereg_20260817.md
REM   Design note:      data/reports/tripletype_design_20260817.md
REM   Model addendum:   data/reports/ab2_addendum_S1H_20260817.md
REM
REM Nothing here touches runs/fpcamp_minfr_hgd569_f125 (CLOSED and merged),
REM runs/fpcamp_minfr_hgd569_f109 (CLOSED), runs/fpcamp_minfr_E1E2_f109
REM (CLOSED), runs/fpcamp_minfr_N1N2_f113 (CLOSED), or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfr_TRIPLE_f125_199.inp ^
  --run-dir runs/fpcamp_minfr_triple_f125 ^
  --no-early-stop ^
  > fpcamp_minfr_triple_f125_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_triple_f125_rc.txt
endlocal
