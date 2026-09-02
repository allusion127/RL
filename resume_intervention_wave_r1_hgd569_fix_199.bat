@echo off
REM ===========================================================================
REM HGD569_f125 CORRECTION -- STAGE 2 of 2: re-evaluate all 160 CHILDREN under
REM the fixed resolver.  RUN diag_intervention_wave_r1_hgd569_fix_199.bat FIRST
REM and only proceed when its three acceptance checks passed.
REM
REM SCOPE: the 160 CHILDREN only.  The 20 PARENTS are NOT re-run -- they were
REM labelled by the campaigns (fpcamp_minfr_hgd569_f125 / _seedctl), which pass
REM resolver= and were never on the defective path (memo sec.5.1).
REM
REM WHY A NEW RUN DIR: runs\intervention_wave_r1 is the round-1 evidence and its
REM ablation_results.jsonl is what ablation_wave._done() resumes from -- running
REM into it would make every chain look "already settled" and skip all 160.  A
REM fresh dir starts with an empty jsonl, so the plan's 160 record_ids are all
REM todo.  Round 1's dir is not read, not written and not deleted.
REM
REM WHY --campaign-suffix _v2: record_id is sha256(pattern|library|pair|knobs)
REM and does NOT hash the campaign, so these 160 rows carry the SAME record_ids
REM as the quarantined ones.  On merge-store dedup_upsert keeps the higher
REM _quality_rank (converged*8 + valid*4 + flat*2 + fxy): the quarantined rows
REM are valid=False (rank <= 11) and these are converged+valid+mapped+F_xy
REM (rank 15), so the corrected label UPGRADES the quarantined one IN PLACE and
REM the surviving row's `campaign` column reads intervention_HGD569_f125_v2.
REM That is the whole point of quarantining rather than deleting.
REM
REM --allow-fallback IS REQUIRED (pair_ecore:MAS_RST.APRQ_11_0705.02, level 3 --
REM the cell has no native restart and its parents carry the same one).
REM check_restart still refuses any drift off the parents' restart.
REM
REM RESUME: run THIS FILE again.  The cell resumes from its own
REM ablation_results.jsonl in the fix run dir.
REM
REM Budget: 160 chains at the measured 199 cadence (24 workers) ~= 1.8 h.
REM Nothing here touches runs\intervention_wave_r1, runs\produce_fxyera_r1, any
REM other runs\ dir, or boxes 198 / 181 / 238.
REM ===========================================================================
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
set PLAN=data/design/intervention_wave_r1.json
set RUNDIR=runs/intervention_wave_r1_hgd569_fix
set LOG=intervention_wave_r1_hgd569_fix_out.log

echo ==== HGD569_f125 fix RUN (160 children, campaign _v2) ==== >> %LOG%
"%PY%" -u intervention_wave.py run ^
  --plan %PLAN% ^
  --kit paramA ^
  --cells HGD569_f125 ^
  --package data/design/package ^
  --fuel-types data/store/fuel_types.parquet ^
  --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir %RUNDIR% ^
  --campaign-suffix _v2 ^
  --workers 24 --host-reserve 1 --wave-size 24 --max-cycles 16 ^
  --allow-fallback ^
  >> %LOG% 2>&1
set RC_RUN=%ERRORLEVEL%

echo ==== HGD569_f125 fix KIT (campaign intervention_HGD569_f125_v2) ==== >> %LOG%
"%PY%" -u intervention_wave.py kit ^
  --plan %PLAN% ^
  --kit paramA ^
  --cells HGD569_f125 ^
  --package data/design/package ^
  --fuel-types data/store/fuel_types.parquet ^
  --run-dir %RUNDIR% ^
  --campaign-suffix _v2 ^
  >> %LOG% 2>&1
set RC_KIT=%ERRORLEVEL%

echo run=%RC_RUN% kit=%RC_KIT% > intervention_wave_r1_hgd569_fix_rc.txt
echo run=%RC_RUN% kit=%RC_KIT%
echo.
echo Ship back: %RUNDIR%\intervention_HGD569_f125\kitdata  (merge-store on the
echo coordinator; the 160 rows UPGRADE the quarantined ones by record_id).
endlocal
