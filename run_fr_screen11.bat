@echo off
REM STEP 2 of data/reports/kcurve_fusion_memo_20260809.md: complete the ga80
REM fixed-pattern fuel screen.  The 11 remaining native-restart pairs of
REM FEASIBLE_PACKAGE/bases, each loaded in natural role order onto the CURRENT
REM flattest core via fr_transfer --select flat --k 1 -- the same construction
REM as arms A0-C4, one equilibrium chain per pair.  NOTE: --select flat re-picks
REM rank 0 from the store at launch time; the 2026-08-10 run anchored on
REM be375a5f8790 (node_peak 1.1932), NOT the older b0ff11ef16de (1.2085),
REM because the fpcamp3 merge had landed first.  The jsonl is authoritative.
REM
REM Purpose: takes the dose-regression n from 9 to 20.  At n=9 the Bonferroni
REM threshold over ~30 candidate predictors is |r| >= 0.882 and only
REM cyclen ~ kinf30 passes; at n=20 it drops to ~0.65 and node_peak predictors
REM (the k-radial-gradient family, the ADF block) become testable for the
REM first time.  Sequential on purpose: one chain in flight keeps the box
REM responsive for the v520 batch that will land alongside.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
del runs_fr_screen11.log 2>nul
for %%P in (G3_G4 H1_H2 J3_J4 J5_J6 K1_K2 K3_K4 L1_L2 L5_L6 N1_N2 N3_N4 N5_N6) do (
  echo ==== %%P ==== >> runs_fr_screen11.log
  python -u fr_transfer.py --target-pair %%P --select flat --k 1 ^
    --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
    --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
    --run-dir runs/fr_screen11/%%P --workers 4 --max-cycles 16 >> runs_fr_screen11.log 2>&1
)
echo done > runs_fr_screen11_rc.txt
endlocal
