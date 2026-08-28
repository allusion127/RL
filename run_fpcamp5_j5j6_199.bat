@echo off
REM ============================================================================
REM Frontier campaign v5 on 199 -- FIRST flat_power campaign in a NEW CELL:
REM   J5_J6 / feed 121 / e_core 5.10 (ga80), champion 20260810_bu_T (cond v6b).
REM   100 MASTER calls, 12 waves x 8 + 4 reserve.  Full justification, all the
REM   measured numbers and the knob audit are in fpcamp5_j5j6_199.inp's header.
REM
REM READ THIS BEFORE JUDGING WAVE 0: this cell is NOT flatter than E1_E2.  The
REM 31 flattest E1_E2/f121 patterns transferred into J5_J6 with the arrangement
REM held fixed lost 0.0888 of node_peak (31/31) while gaining 0.0253 of F_r,
REM 36.9 ppm of CBC and 1.96 EFPD of cycle (31/31 each).  The comparator for
REM this run is the J5_J6 incumbent node_peak 1.2561, NOT the program frontier
REM 1.1899.  Anything at or below 1.2561 is progress.
REM
REM ---------------------------------------------------------------------------
REM SHIP LIST -- to C:\Users\USER\lpopt_work\kit_frontier on 199
REM
REM  0. FIRST, BEFORE ANYTHING: pull 199's kit store and merge it home, THEN
REM     re-ship the merged records.parquet.  Shipping over an un-merged kit
REM     store destroys whatever fpcamp4_199 has produced since the last pull.
REM         python -m lpopt merge-store --kit-data <pulled 199 data folder>
REM
REM  #  item                                   blocking?  why
REM  -- -------------------------------------- ---------  --------------------
REM  1  lpopt\  (whole source tree, 6.4 MB)     YES        the 199 kit served
REM        cond_schema v6 for fpcamp v1..v4.  v6b landed locally 2026-08-10
REM        (lpopt\model\featurize.py CHANNELS_V6B, 58 ch = v6's 52 + the
REM        source-chain burn-state block).  Without it the deck cannot even
REM        LOAD: config.py:1462 sources _valid_cond_schemas() from
REM        featurize.CHANNELS_BY_SCHEMA, so an older kit raises
REM        "ConfigError: [model] cond_schema 'v6b' invalid".  And if it did
REM        load, model_api.py:654/663 would refuse the 58-channel checkpoint
REM        against a 52-channel encoder.  It fails LOUDLY either way, never
REM        silently -- but it fails, so ship it.
REM  2  data\models\20260810_bu_T\  (199 MB)     YES        the serving champion
REM        (5 members x model.pt + meta.json, plus its 6 calibration artifacts:
REM        calibration / cell_calibration / f_r_calibration / f_q_calibration /
REM        cbc_calibration / ao_abs_calibration / flatness_calibration).
REM        Ship the WHOLE directory -- the map_calibration precondition is
REM        decided on a SHA1 over every member's meta.json AND model.pt
REM        (map_calibration.py:180-219), so a partial copy fails the check.
REM  3  data\store\map_calibration.json (9,228 B) YES      THE PRECONDITION.
REM        Refit locally 2026-08-10T07:06:44 against 20260810_bu_T (fingerprint
REM        9048108754f9..., split S1, fold C, 1,280 rows, 7 cells).  The kit's
REM        current copy was fitted on an OLDER champion, so leaving it in place
REM        makes the campaign ABORT at construction with ModelMismatchError
REM        (campaign.py:518-520).  sha256 6237f8f25485921a...
REM  4  data\store\records.parquet (21,069,795 B) YES      the elite parent pool.
REM        The 31 `frtransfer_J5J6boot` rows exist ONLY in the canonical store.
REM        Without them _store_elites finds 23 J5_J6 rows of which 22 carry NO
REM        node_peak, the 32-parent elite arm collapses and 65% of the pool
REM        degrades toward random.  See #0 -- merge before shipping over.
REM  5  data\store\flat_scale.json (8,726 B)      verify    UNCHANGED (fit
REM        2026-07-26, sha256 4575f31144eefee5...).  The deck's
REM        flatpower_per_cell_scale = true needs cell `feed=121|ebin=5.05`
REM        (n=31, peak 0.151892 / cov 0.057316) to be present.  If the kit's
REM        copy already hashes the same, skip the transfer; if it differs at
REM        all, ship this one -- a different flat_scale silently redefines the
REM        objective's units.
REM  6  fpcamp5_j5j6_199.inp                      YES
REM  7  run_fpcamp5_j5j6_199.bat  (this file)     YES
REM  8  data\store\maps.npz (184 MB)              NO        the campaign only
REM        APPENDS to the kit's own copy (campaign.py:1735 write_maps
REM        append=True); the objective reads the node_peak / map_cov COLUMNS of
REM        records.parquet, not the raw maps.  Skip the transfer.
REM  9  data\splits\S1.json                       NO        only a map_calibration
REM        refit needs it, and the refit is already done (#3).
REM 10  FEASIBLE_PACKAGE                          already there -- 199 ran the
REM        J5_J6 transfer out of it today.  bases\J5_J6\MAS_RST.APRQ_11_0638.15
REM        + 22 exact decks under cores\J5_J6\.  Verified by `lpopt check`
REM        against the identical local package: 526 PASS / 0 FAIL.
REM
REM ---------------------------------------------------------------------------
REM NO map_calibration REFIT STEP HERE, AND THIS TIME FOR A DIFFERENT REASON.
REM   run_fpcamp_199.bat (v1) refit the artifact on the box.  v2/v3/v4 skipped
REM   the refit because the shipped S1 covered the whole store and fold C was
REM   empty.  That is NO LONGER TRUE -- the store has grown 1,444 rows past the
REM   frozen S1 id list -- so the refit RAN, locally, on 1,280 fold-C rows, and
REM   its output ships as item #3.  Nothing to run on the box.
REM
REM ---------------------------------------------------------------------------
REM PRE-LAUNCH SANITY (30 s, runs no MASTER, starts nothing):
REM     cd /d C:\Users\USER\lpopt_work\kit_frontier
REM     C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe -m lpopt check ^
REM         --input fpcamp5_j5j6_199.inp
REM   Must print "RESULT: OK".  A "[model] cond_schema 'v6b' invalid" here means
REM   ship item #1 (lpopt\) and re-run.
REM
REM EXPECTED IN THE FIRST 60 SECONDS OF THE LOG -- these lines are the proof the
REM   preconditions resolved, copied from the local dry-run of the same deck:
REM     [optimize] flat_power objective = -(node_peak/0.1519 + 0.5 * map_cov/0.0573)
REM         | cell feed=121|ebin=5.05 scales FITTED (...flat_scale.json)
REM         | F_r SAFETY GATE at 1.700 (bias-corrected per cell, not an objective)
REM     [optimize] map_calibration (...map_calibration.json, model 20260810_bu_T):
REM         node_peak bias -0.004257 sigma_extra 0.0 [cell] | map_cov bias
REM         -0.004622 sigma_extra 0.0 [cell]
REM   If "model UNDECLARED", "GLOBAL fallback" or "HELD -- no map-head bias
REM   correction available" appears instead, an artifact did not ship: STOP.
REM
REM ---------------------------------------------------------------------------
REM DELIVERY POLICY (2026-08-10, retrospective, from the verify-5-of-20 K2
REM   result): the delivered candidate set is the TOP-5 of delivery.json's
REM   `ranked` array, not `ranked[0]`.  The campaign writes the FULL ranking
REM   (campaign.py:2149 passes no top_k), so this is a read-time rule; the deck
REM   raises [constraints] post_verify_top_k 3 -> 5 so the D9 SDM/MTC gate
REM   verifies exactly that set (2 extra MASTER calls, reported separately from
REM   the 100-call search budget).
REM
REM ---------------------------------------------------------------------------
REM chcp 65001 + PYTHONUTF8 + PYTHONIOENCODING ARE MANDATORY.  The flat_power
REM construction log carries em-dashes and the section sign (program "2.1"
REM citations); on a cp949 console print() raises UnicodeEncodeError inside
REM CampaignDriver.__init__ and the run dies before wave 0 (observed
REM 2026-08-01, exit 90).
REM ============================================================================
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
set PY=C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe
"%PY%" -u -m lpopt optimize --input fpcamp5_j5j6_199.inp --run-dir runs/fpcamp5_j5j6 --no-early-stop > fpcamp5_j5j6_199_out.log 2>&1
echo %ERRORLEVEL% > fpcamp5_j5j6_199_rc.txt
endlocal
