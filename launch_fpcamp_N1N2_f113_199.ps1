# launch_fpcamp_N1N2_f113_199.ps1 -- runs ON 199.  Arms the LOW-FEED OPENING
# campaign (N1_N2 / feed 113 / ga80, pure F_r minimisation, no cycle band).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_N1N2_f113_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe.  schtasks is documented to no-op silently on this fleet
# (data/reports/ab2_addendum_ADF_20260810.md section 3), so it is not used.
#
# BUSY GATE (v520_b2 precedent): REFUSES rather than stacks.  199 was idle at
# audit time; if anything is running, that assumption is wrong and this must not
# add a second MASTER queue to the box.
#
# DECK GATE: the deck must be the one the pre-registration hashed.  A wrong or
# truncated deck is cheaper to catch here than 100 MASTER calls later.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate ---------------------------------------------------------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("F113 REFUSED: box busy (lpopt_python=$busy master=$mc)"); exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'fpcamp_minfr_N1N2_f113_199.inp'
if (-not (Test-Path $deck)) { Write-Output "F113 REFUSED: deck not found ($deck)"; exit 1 }
$want = 'D93ECFF3343959762B6EF80DD067EA98F14CD3901D903E2792E32D0E8782734D'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "F113 REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

# -- precondition gate: the champion and the refreshed store must be present --
$m = Join-Path $k 'data\models\s1f\ensemble.json'
if (-not (Test-Path $m)) { Write-Output "F113 REFUSED: data\models\s1f not on the kit"; exit 1 }
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "F113 REFUSED: store missing"; exit 1 }
$stLen = (Get-Item $st).Length
if ($stLen -ne 21942421) {
  Write-Output "F113 REFUSED: store is not the refreshed copy (bytes=$stLen, want 21942421)"; exit 1
}
$bat = Join-Path $k 'run_fpcamp_minfr_N1N2_f113_199.bat'
if (-not (Test-Path $bat)) { Write-Output "F113 REFUSED: run bat not at the kit root"; exit 1 }
# ga80 package + the pair's base restart (level-2 pair_feed source)
$base = Join-Path $k 'FEASIBLE_PACKAGE\bases\N1_N2'
if (-not (Test-Path $base)) { Write-Output "F113 REFUSED: FEASIBLE_PACKAGE\bases\N1_N2 missing"; exit 1 }

# -- FRESH run-dir -----------------------------------------------------------
# A stale partial would be read back as "already done" by the state.json /
# labels.jsonl resume and would silently shrink the campaign.
Remove-Item (Join-Path $k 'runs\fpcamp_minfr_N1N2_f113') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_N1N2_f113_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_N1N2_f113_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("F113 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "F113 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 45
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_minfr_N1N2_f113' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_minfr_N1N2_f113_out.log'
Write-Output ("F113 armed: deck_sha_ok=True python=$n master=$mc2" +
              "  log=$log  rc=$k\fpcamp_minfr_N1N2_f113_rc.txt")
if (Test-Path $log) {
  Write-Output '--- log header ---'
  Get-Content $log -TotalCount 45
} else {
  Write-Output '--- log not created yet ---'
}
