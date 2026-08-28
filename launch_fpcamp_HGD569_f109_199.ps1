# launch_fpcamp_HGD569_f109_199.ps1 -- runs ON 199.  Arms the HIGH-Gd
# boron-opened-cell F_r assault (P6253Z1G06N24_P6253Z2G10N24 / feed 109 / paramA,
# e_core 5.694, n_gd 24).  Pure F_r minimisation under five gates, no cycle band.
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_HGD569_f109_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet).
#
# BUSY GATE: REFUSES rather than stacks.
# DECK GATE: the deck must be the one the pre-registration hashed.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate (includes meshcalc / scoping-mesh / prior-campaign leftovers) --
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("HGD569 REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'fpcamp_minfr_HGD569_f109_199.inp'
if (-not (Test-Path $deck)) { Write-Output "HGD569 REFUSED: deck not found ($deck)"; exit 1 }
$want = '244384F8DF986BA048FACB5A15911E88B28F1BF89FA8250D36B081A334564509'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "HGD569 REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

# -- precondition gates ------------------------------------------------------
if (-not (Test-Path (Join-Path $k 'data\models\s1g\ensemble.json'))) {
  Write-Output "HGD569 REFUSED: data\models\s1g not on the kit"; exit 1 }
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "HGD569 REFUSED: store missing"; exit 1 }
$bat = Join-Path $k 'run_fpcamp_minfr_HGD569_f109_199.bat'
if (-not (Test-Path $bat)) { Write-Output "HGD569 REFUSED: run bat not at the kit root"; exit 1 }
# paramA routing: the design package, NOT FEASIBLE_PACKAGE
$pkg = Join-Path $k 'data\design\package'
if (-not (Test-Path (Join-Path $pkg 'registry.json'))) {
  Write-Output "HGD569 REFUSED: data\design\package\registry.json missing (paramA routing)"; exit 1 }
# level-3 pair_ecore restart source, proven by all 16 anchor rows at this cell
$base = Join-Path $pkg 'bases\P0_P1\MAS_RST.APRQ_11_0705.02'
if (-not (Test-Path $base)) {
  Write-Output "HGD569 REFUSED: bases\P0_P1\MAS_RST.APRQ_11_0705.02 missing (expected level-3 restart source)"; exit 1 }

# -- FRESH run-dir -----------------------------------------------------------
Remove-Item (Join-Path $k 'runs\fpcamp_minfr_hgd569_f109') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_hgd569_f109_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_hgd569_f109_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("HGD569 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "HGD569 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 60
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_minfr_HGD569_f109' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_minfr_hgd569_f109_out.log'
Write-Output ("HGD569 armed: deck_sha_ok=True python=$n master=$mc2  log=$log")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 45 }
else { Write-Output '--- log not created yet ---' }
