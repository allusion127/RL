# launch_fpcamp_E1E2_f109_199.ps1 -- runs ON 199.  Arms the LOW-FEED OPENING
# campaign round 2 (E1_E2 / feed 109 / ga80, pure F_r minimisation, no cycle band).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_E1E2_f109_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet).
#
# BUSY GATE: REFUSES rather than stacks.  Also refuses if a lingering meshcalc /
# scoping-mesh python is still holding the box.
# DECK GATE: the deck must be the one the pre-registration hashed.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate (includes meshcalc / scoping-mesh leftovers) ------------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("F109 REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'fpcamp_minfr_E1E2_f109_199.inp'
if (-not (Test-Path $deck)) { Write-Output "F109 REFUSED: deck not found ($deck)"; exit 1 }
$want = '1A967B37A93B57F9DC93E7B2A8F2F23422ED740C9299E16D0196D026392AD247'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "F109 REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

# -- precondition gates ------------------------------------------------------
if (-not (Test-Path (Join-Path $k 'data\models\s1g\ensemble.json'))) {
  Write-Output "F109 REFUSED: data\models\s1g not on the kit"; exit 1 }
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "F109 REFUSED: store missing"; exit 1 }
$bat = Join-Path $k 'run_fpcamp_minfr_E1E2_f109_199.bat'
if (-not (Test-Path $bat)) { Write-Output "F109 REFUSED: run bat not at the kit root"; exit 1 }
# level-2 pair_feed restart source: nearest same-pair folder to feed 109 is f117
$base = Join-Path $k 'FEASIBLE_PACKAGE\bases\E1_E2_f117'
if (-not (Test-Path $base)) {
  Write-Output "F109 REFUSED: FEASIBLE_PACKAGE\bases\E1_E2_f117 missing (expected level-2 restart source)"; exit 1 }

# -- FRESH run-dir -----------------------------------------------------------
Remove-Item (Join-Path $k 'runs\fpcamp_minfr_E1E2_f109') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_E1E2_f109_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_E1E2_f109_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("F109 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "F109 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 45
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_minfr_E1E2_f109' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_minfr_E1E2_f109_out.log'
Write-Output ("F109 armed: deck_sha_ok=True python=$n master=$mc2  log=$log")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 45 }
else { Write-Output '--- log not created yet ---' }
