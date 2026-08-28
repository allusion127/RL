@echo off
REM verify-5-of-20 BATCH 3 -- THE DECIDING BATCH.  Runs on the LOCAL box.
REM
REM Pre-registration : data/reports/v520_preregistration_20260810.md
REM Batch-2 addendum : data/reports/v520_addendum_b2_20260810.md   (seed 4784, box 198)
REM Batch-3 addendum : data/reports/v520_addendum_b3_20260810.md   (seed 4785, THIS run)
REM
REM WHERE IT STANDS:
REM   batch 1  local / seed 4636   P1 1.4999  P5 1.4928  ORACLE 1.4784  P1-P5 +0.0071  REFUTES
REM   batch 2  198   / seed 4784   P1 1.5769  P5 1.4985  ORACLE 1.4797  P1-P5 +0.0784  SUPPORTS
REM   -> UNREPLICATED (b2 addendum sec. 5.4); mean so far 0.0428; third batch required.
REM
REM Seed 4785 is the NEXT VALUE IN THE LADDER REGISTERED IN THE B2 ADDENDUM sec. 3.1
REM ("4784, 4785, 4786, ... consumed in ascending order"), not a seed chosen after
REM seeing two results.
REM
REM THREE-BATCH DECISION RULE (b3 addendum sec. 5.1) -- fixed before this run:
REM   SUPPORTS K2  iff mean(P1-P5) >= 0.030 AND >= 2 of 3 batches have P1-P5 >= 0.010
REM   REFUTES  K2  iff mean(P1-P5) <  0.010 AND >= 2 of 3 batches have P1-P5 <  0.010
REM   UNRESOLVED at n=3 otherwise.
REM Per-batch thresholds are UNCHANGED (0.030 / 0.010) and are NOT re-tuned.
REM
REM POWER DISCLOSURE, registered before the draw (b3 addendum sec. 5.2): with b1+b2
REM summing to 0.0855 and P1-P5 >= 0 by construction, an overall REFUTES is
REM ARITHMETICALLY UNREACHABLE.  This batch decides exactly between SUPPORTS
REM (b3 >= 0.010) and UNRESOLVED (b3 < 0.010), and nothing else.
REM
REM MANDATORY CO-REPORT (b3 addendum sec. 6): the SUPPORTS bar 0.030 sits BELOW the
REM measured random-ranking null (0.0404 b1 / 0.0242 b2, mean 0.0323).  A ZERO-SKILL
REM ranker would "SUPPORT K2" under the rule.  The verdict text MUST carry the nulls
REM and the mean excess-over-null, and if that excess is <= 0 it MAY NOT claim the
REM verification policy beats model upgrades.
REM
REM STOPPING RULE (sec. 5.3): batch 3 is the LAST batch under this registration.
REM An UNRESOLVED outcome buys a DESIGN change, never a batch 4.
REM
REM Candidates must already be frozen:  python v520_gen.py --run-dir runs/v520_b3 ...
REM Resumable: re-running skips candidates already in runs/v520_b3/v520_results.jsonl.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u v520_run.py ^
  --run-dir runs/v520_b3 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --workers 12 --max-cycles 16 --check-novelty > runs_v520_b3.log 2>&1
echo %ERRORLEVEL% > runs_v520_b3_rc.txt
endlocal
