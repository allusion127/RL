@echo off
REM ============================================================================
REM RULE-CONSTRUCT ACID TEST -- 8 MASTER chains.  DO NOT LAUNCH CASUALLY.
REM The ISSUER launches this, not the agent.  Staged 2026-08-11.
REM
REM THE QUESTION (program north star): patterns built FROM THE VALIDATED RULES
REM ALONE (scratchpad rules/validated.json; constructor rule_construct.py --
REM no surrogate in the loop, no search campaign) must approach what the
REM 100-call campaigns found in ga80 | E1_E2 | feed 121.
REM
REM CANDIDATES (frozen BEFORE any chain; gate V5 quotes the sha256):
REM   %LOCALAPPDATA%\Temp\claude\...\scratchpad\rules\acid_batch\candidates.json
REM   8 = 4 per profile (minfr / flat): top-2 by champion prediction (referee
REM   only -- it built nothing) + 2 diverse mid-rank honesty probes.
REM
REM PRE-REGISTERED SUCCESS CRITERIA -- fixed HERE, BEFORE the first chain runs;
REM these numbers do not move:
REM   store records (100-call campaigns): F_r 1.4636, node_peak 1.1899
REM   SUCCESS(minfr): min realized F_r over the 4 minfr candidates <= 1.479
REM                   (record + 0.015), with cbc_max <= 1600, f_q <= 2.41,
REM                   620 <= cyclen <= 645.
REM   SUCCESS(flat) : min realized node_peak over the 4 flat candidates
REM                   <= 1.205 (record + 0.015), with f_r <= 1.55 and the same
REM                   cbc / f_q / cyclen gates.
REM   Either SUCCESS = the explicit rules capture most of what 100-call search
REM   finds on that objective.  Both FAIL is a real, publishable negative.
REM   (cbc gate is the NEW 1600 limit; in-store comparison rows were produced
REM   under 1550 -- report that footnote with the verdict.)
REM
REM VALIDITY GATES (all must hold or the verdict is VOID):
REM   V1 all 8 converged   V2 fallback_level 0 + native restart
REM   V3 8 patterns pairwise distinct   V4 none already in the store
REM   V5 candidates.json sha256 quoted in the verdict
REM
REM SECONDARY (report-only): Spearman(predicted, realized) over the 8 -- the
REM mid-rank honesty probes exist so this number is readable.
REM
REM Resumable: re-running skips candidates already in
REM runs/rule_acid/rule_acid_results.jsonl.
REM Dry-run first:  python rule_acid_run.py --dry-run
REM ============================================================================
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u rule_acid_run.py ^
  --candidates "C:\Users\USER\AppData\Local\Temp\claude\c--Users-USER-Desktop-CT-RPL-2-Project-KNF-LEU------------------2026-2---\8888f052-fa4d-46f0-a439-ef3441b3b061\scratchpad\rules\acid_batch\candidates.json" ^
  --run-dir runs/rule_acid ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --workers 12 --max-cycles 16 > runs_rule_acid.log 2>&1
echo %ERRORLEVEL% > runs_rule_acid_rc.txt
endlocal
