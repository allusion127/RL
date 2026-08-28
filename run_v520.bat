@echo off
REM verify-5-of-20 -- objection K2 of data/reports/kcurve_fusion_memo_20260809.md.
REM Pre-registered decision rule: data/reports/v520_preregistration_20260810.md,
REM written 2026-08-10 BEFORE the candidates were generated and BEFORE any chain ran.
REM
REM   20 novel candidates in ga80 / E1_E2 / feed 121, frozen with the model's
REM   PREDICTED F_r ranking in runs/v520/candidates.json.  ALL 20 verified,
REM   native restart only.  Three policies, scored counterfactually from those
REM   SAME 20 labels (the baseline therefore costs zero extra chains):
REM
REM     P1     realized F_r of predicted-rank 1
REM     P5     min realized F_r over predicted-ranks 1..5
REM     ORACLE min realized F_r over all 20
REM
REM   SUPPORTS K2  :  P1 - P5 >= 0.030
REM   REFUTES  K2  :  P1 - P5 <  0.010
REM   GRAY ZONE    :  otherwise -- report only
REM
REM Stop condition: if this REFUTES, the verification policy is NOT the cheap win
REM and the MODEL is the bottleneck -- the "0.0544 for free" argument against the
REM memo section 3 model upgrades (v7 ADF channels, burnup placement, fusion head)
REM is withdrawn in writing and those proposals recover their priority.
REM
REM Candidates must already be frozen:  python v520_gen.py   (budget 0, no MASTER)
REM Resumable: re-running skips candidates already in runs/v520/v520_results.jsonl.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u v520_run.py ^
  --run-dir runs/v520 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --workers 12 --max-cycles 16 --check-novelty > runs_v520.log 2>&1
echo %ERRORLEVEL% > runs_v520_rc.txt
endlocal
