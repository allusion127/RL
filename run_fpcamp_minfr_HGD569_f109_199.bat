@echo off
REM HIGH-Gd BORON-OPENED CELL, F_r ASSAULT -- P6253Z1G06N24_P6253Z2G10N24 /
REM feed 109 / paramA (e_core 5.694, n_gd 24) on BOX 199.
REM
REM Objective: minimise F_r under the four safety gates (F_r<=1.55, CBC<=1600,
REM F_q<=2.41, |AO|<=0.30) PLUS the new predicted pin-BU gate (<=78 GWd/tU).
REM NO cycle-length target and NO band -- minfr_lambda = 1000 (lambda=0 would
REM invert the objective into cycle-length maximisation).
REM
REM Budget 60 (7 waves x 8 + 4 reserve), NOT 100: both min_fr precedents had a
REM stalled frontier by wave 3-5 (fpcamp_E1E2_f109_results_20260817.md sec 8.4).
REM
REM Why this cell: the mesh-v3 anchors measured CBC floor 1405.2 ppm here with
REM 16/16 sampler cores under the 1600 limit -- the widest boron opening in the
REM whole high-enrichment set.  Boron has stopped binding; F_r now binds.
REM   data/reports/mesh_v3_20260817/README.md sec 5e-2 / 5e-4
REM
REM Pre-registration: data/reports/fpcamp_HGD569_f109_prereg_20260817.md
REM plus the deck header (fpcamp_minfr_HGD569_f109_199.inp).
REM
REM Preconditions shipped 2026-08-17 before arming:
REM   * lpopt/ source with the min_fr predicted pin-BU gate
REM   * data/store/records.parquet -- post-meshv3-anchor canonical (74,357 rows),
REM     which is what carries this pair's 47 rows at all
REM
REM Nothing here touches runs/fpcamp_minfr_E1E2_f109 (CLOSED and merged),
REM runs/fpcamp_minfr_N1N2_f113 (CLOSED), or boxes 198 / 181 / 238.
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize ^
  --input fpcamp_minfr_HGD569_f109_199.inp ^
  --run-dir runs/fpcamp_minfr_hgd569_f109 ^
  --no-early-stop ^
  > fpcamp_minfr_hgd569_f109_out.log 2>&1
echo %ERRORLEVEL% > fpcamp_minfr_hgd569_f109_rc.txt
endlocal
