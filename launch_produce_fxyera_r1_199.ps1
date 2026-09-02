# launch_produce_fxyera_r1_199.ps1 -- runs ON 199.  Arms the F_xy-ERA
# COVERAGE-FILL PRODUCE ROUND 1: paramA, 19 strata x n_target 40, objective
# neutral (DoE), MAS_OUT of every converged chain RETAINED so the F_xy parser
# can retro-label it.  This is the programme's first f_xy label batch.
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_produce_fxyera_r1_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet -- same as the TRIPLE r2 family).
#
# GATES, all REFUSE rather than stack or run on the wrong inputs:
#   BUSY    no master4.0m4_r1, no lpopt/ablation/batchswap/mesh python.
#   DECK    sha256 must equal the value the pre-registration froze.
#   STORE   sha256 of records.parquet must equal the 2026-08-29 canonical store
#           -- a stale store starves the arm-C elite pools AND mis-dedups.
#   FUEL    sha256 of fuel_types.parquet -- without it the level-3 pair_ecore
#           restart fallback is disabled and 13 of 19 strata die.
#   ROUTING data\design\package registry + the 7 restarts val_assets.py resolved.
#   DISK    >= 30 GB free.  Retention costs ~9.5 MB/chain x ~1120 chains ~ 11 GB.
#   MODEL   data\models\s1i present.  NOT schema-gated: `produce` has no
#           acquisition and never loads the checkpoint (the [model] block only
#           supplies library_id = paramA, the resolver routing key).
#
#   Pre-registration: data/reports/produce_fxyera_r1_prereg_20260829.md
#
# Nothing here touches runs/fpcamp_minfr_triple_f125_r2 (TRIPLE r2), any other
# runs/ dir, or boxes 198 / 181 / 238.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate ---------------------------------------------------------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("FXYERA1 REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'produce_fxyera_r1_199.inp'
if (-not (Test-Path $deck)) { Write-Output "FXYERA1 REFUSED: deck not found ($deck)"; exit 1 }
$want = '6B72DDA2DFDA3124327CEB3B17D07EBB5B58281426BD7C6F15B5ABB7C57CD380'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "FXYERA1 REFUSED: deck sha256 mismatch (a remote edit adds a UTF-8 BOM and breaks the TOML)"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

# -- store gate: the canonical 2026-08-29 store -------------------------------
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "FXYERA1 REFUSED: data\store\records.parquet missing"; exit 1 }
$wantStore = 'D2196B5EC0F53D59432DA071DC063CA35FB54BA832BA2A0B0356A5D9535F4B0F'
$gotStore  = (Get-FileHash -Algorithm SHA256 $st).Hash
if ($gotStore -ne $wantStore) {
  Write-Output "FXYERA1 REFUSED: store sha256 mismatch -- ship data\store\records.parquet from the coordinator"
  Write-Output "  expected $wantStore"
  Write-Output "  found    $gotStore"
  exit 1
}

# -- fuel table gate: level-3 pair_ecore fallback depends on it ---------------
$ft = Join-Path $k 'data\store\fuel_types.parquet'
if (-not (Test-Path $ft)) { Write-Output "FXYERA1 REFUSED: data\store\fuel_types.parquet missing (level-3 restart fallback disabled)"; exit 1 }
$wantFuel = 'FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137'
$gotFuel  = (Get-FileHash -Algorithm SHA256 $ft).Hash
if ($gotFuel -ne $wantFuel) {
  Write-Output "FXYERA1 REFUSED: fuel_types.parquet sha256 mismatch"
  Write-Output "  expected $wantFuel"
  Write-Output "  found    $gotFuel"
  exit 1
}

# -- model presence (NOT schema-gated: produce never loads it) ----------------
if (-not (Test-Path (Join-Path $k 'data\models\s1i\ensemble.json'))) {
  Write-Output "FXYERA1 REFUSED: data\models\s1i not on the kit"; exit 1 }

# -- paramA routing: the design package, NOT FEASIBLE_PACKAGE ------------------
$pkg = Join-Path $k 'data\design\package'
if (-not (Test-Path (Join-Path $pkg 'registry.json'))) {
  Write-Output "FXYERA1 REFUSED: data\design\package\registry.json missing (paramA routing)"; exit 1 }
# The exact restarts `python val_assets.py produce_fxyera_r1_199.inp` resolved
# (19/19, RESULT resolved=19 failed=0) on the coordinator.
$needed = @(
  'bases\P0_P1\MAS_RST.APRQ_11_0705.02',
  'bases\Q1_Q2\MAS_RST.APRQ_11_0767.14',
  'bases\Q7_Q8\MAS_RST.APRQ_12_0777.14',
  'bases\T6_T4\MAS_RST.APRQ_10_0615.11',
  'bases\T5_T6\MAS_RST.APRQ_11_0632.51',
  'bases\T1_T4_f117\MAS_RST.APRQ_12_0635.61',
  'bases\T3_T4\MAS_RST.APRQ_11_0578.27'
)
foreach ($n in $needed) {
  if (-not (Test-Path (Join-Path $pkg $n))) {
    Write-Output "FXYERA1 REFUSED: $n missing (a stratum would resolve unresolved)"; exit 1 }
}

# -- disk gate: retained MAS_OUT is the whole point, and it costs GB ----------
$drive = (Get-Item $k).PSDrive.Name
$freeGB = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
if ($freeGB -lt 30) {
  Write-Output "FXYERA1 REFUSED: only $freeGB GB free on ${drive}: (need >= 30 GB; ~11 GB of retained MAS_OUT plus headroom)"; exit 1 }

$bat = Join-Path $k 'run_produce_fxyera_r1_199.bat'
if (-not (Test-Path $bat)) { Write-Output "FXYERA1 REFUSED: run bat not at the kit root"; exit 1 }

# -- FRESH run-dir -----------------------------------------------------------
Remove-Item (Join-Path $k 'runs\produce_fxyera_r1') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'produce_fxyera_r1_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'produce_fxyera_r1_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("FXYERA1 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "FXYERA1 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'produce_fxyera_r1' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'produce_fxyera_r1_out.log'
Write-Output ("FXYERA1 armed: deck_sha_ok=True store_sha_ok=True fuel_sha_ok=True freeGB=$freeGB python=$n master=$mc2  log=$log")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 60 }
else { Write-Output '--- log not created yet ---' }
