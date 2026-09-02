@echo off
REM ===========================================================================
REM HGD569_f125 CORRECTION -- STAGE 1 of 2: the 2-chain DIAGNOSTIC.  RUN THIS
REM FIRST.  Do not launch resume_intervention_wave_r1_hgd569_fix_199.bat until
REM both checks below print the expected answer.
REM
REM WHAT IS BEING CORRECTED
REM   data/reports/hgd569_degeneracy_memo_20260830.md.  Round 1's deck-emission
REM   path held a CaseAssetResolver with an EMPTY type_id->alias bridge, so
REM   %%LPD_SHF carried raw 13-char type_ids (F P6253Z1G06N24) instead of the
REM   deck's 2-char %%LPD_B&C aliases (F S3 / F S5).  MASTER absorbed them
REM   silently as a single unrelated batch P6: every one of the 160 chains was
REM   computed on a single-composition core nobody designed (+35 EFPD).
REM
REM THE FIX THIS RUN DEPENDS ON (copy these files to the box BEFORE running)
REM   lpopt/search/assets.py    CaseAssetResolver derives the alias bridge from
REM                             <package>/registry.json; validate_reload_deck now
REM                             refuses any %%LPD_SHF fresh id that is not a
REM                             2-char id present in %%LPD_B&C.
REM   lpopt/search/verify.py    WaveVerifier builds its fallback resolver FROM
REM                             package_root (never the "." sentinel).
REM   lpopt/search/resolver.py  one reader of the bridge.
REM   intervention_wave.py      passes resolver= into the verifier explicitly and
REM                             refuses a paramA cell with no alias bridge; adds
REM                             --campaign-suffix.
REM   (ablation_wave.py is sha-pinned and is NOT modified.)
REM
REM COST: 2 chains, ~0.03 h.  Writes ONLY runs\intervention_wave_r1_hgd569_fix.
REM Round 1's runs\intervention_wave_r1 is left exactly as it is (it is the
REM evidence for the memo).
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

REM --allow-fallback IS REQUIRED and is not a relaxation: HGD569_f125 has no
REM native restart and its 20 parents were themselves labelled at
REM fallback_level 3 on pair_ecore:MAS_RST.APRQ_11_0705.02.  The real guard is
REM intervention_wave.check_restart, which refuses any run whose restart differs
REM from the parents' -- it stays armed and is NOT overridden here.
echo ==== HGD569_f125 fix DIAGNOSTIC (2 chains) ==== >> %LOG%
"%PY%" -u intervention_wave.py run ^
  --plan %PLAN% ^
  --kit paramA ^
  --cells HGD569_f125 ^
  --package data/design/package ^
  --fuel-types data/store/fuel_types.parquet ^
  --exe C:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir %RUNDIR% ^
  --campaign-suffix _v2 ^
  --workers 2 --host-reserve 1 --wave-size 2 --max-cycles 16 ^
  --max-chains 2 ^
  --allow-fallback ^
  >> %LOG% 2>&1
set RC=%ERRORLEVEL%
echo diag=%RC% > intervention_wave_r1_hgd569_fix_rc.txt
if not "%RC%"=="0" (
  echo DIAGNOSTIC FAILED rc=%RC% -- read %LOG%.  Do NOT run the full correction.
  goto :end
)

echo.
echo ================= CHECK 1 -- the EMITTED DECK (decisive, no MASTER) =======
echo Expect: only "F S3" / "F S5" fresh cards, and ZERO hits for the raw type_id.
echo A single hit on the raw id means the fix did not reach this box.
powershell -NoProfile -Command ^
  "$d = Get-ChildItem -Recurse -Filter 'MAS_INP_cy*.inp' '%RUNDIR%\intervention_HGD569_f125\produce_cases'; foreach ($f in $d) { $t = Get-Content -Raw $f.FullName; $shf = [regex]::Match($t, '(?s)%%LPD_SHF.*?\r?\n(?<b>.*?)\r?\n\s*%%').Groups['b'].Value; $ids = [regex]::Matches($shf, '\bF\s+(\S+)\s+\d') ^| ForEach-Object { $_.Groups[1].Value } ^| Sort-Object -Unique; Write-Host ($f.Directory.Name + '  fresh ids: ' + ($ids -join ',')) }"

echo.
echo ================= CHECK 2 -- MASTER's OWN echo ============================
echo Expect an S3/S5 MIX in every row of "== CORE LOADING PATTERN".
echo An all-P6 map is the round-1 defect reproducing: STOP.
powershell -NoProfile -Command ^
  "Get-ChildItem -Recurse -Filter 'MAS_OUT' '%RUNDIR%\intervention_HGD569_f125\master_work' ^| ForEach-Object { Write-Host ('--- ' + $_.FullName); Select-String -Path $_.FullName -Pattern 'CORE LOADING PATTERN' -Context 0,12 }"

echo.
echo ACCEPTANCE (all three must hold before stage 2):
echo   1. CHECK 1 prints only S3 and S5, never P6253Z1G06N24 / P6253Z2G10N24;
echo   2. CHECK 2 prints an S3/S5 mix, never an all-P6 map;
echo   3. the 2 chains' cyclen in %RUNDIR%\intervention_HGD569_f125\ablation_results.jsonl
echo      lands in the cell's REAL band (~708-741 EFPD), not round 1's 762-771.
:end
endlocal
