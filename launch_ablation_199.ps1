# launch_ablation_199.ps1 -- runs ON 199.  Arms the 1-move ablation wave.
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_ablation_199.ps1"
#
# Pre-registration: data/reports/ablation_wave_prereg_20260815.md
#
# BUSY GATE (v520_b2 precedent): refuses rather than stacks.  199 was freed by
# the T6_T4 r8 close-out; if anything is running, that assumption is wrong and
# this must not add a second MASTER queue to the box.
#
# PLAN GATE: the manifest must be the one the pre-registration hashed.  A wrong
# or truncated plan is cheaper to catch here than 150 chains later.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("ABLATION REFUSED: box busy (lpopt_python=$busy master=$mc)"); exit 1
}

$plan = Join-Path $k 'data\design\ablation_wave_20260815.json'
if (-not (Test-Path $plan)) {
  Write-Output "ABLATION REFUSED: $plan not found (scp it first)"; exit 1
}
$want = '0693DEF74A2EC7897AD764721EDCE37850607E42DFB5B51728C7D6270433B6C4'
$got = (Get-FileHash -Algorithm SHA256 $plan).Hash
if ($got -ne $want) {
  Write-Output ("ABLATION REFUSED: plan sha256 mismatch (pre-registration §8)")
  Write-Output ("  expected $want")
  Write-Output ("  found    $got")
  exit 1
}
foreach ($f in @('ablation_wave.py', 'run_ablation_199.bat')) {
  if (-not (Test-Path (Join-Path $k $f))) {
    Write-Output "ABLATION REFUSED: $f not at the kit root"; exit 1
  }
}

# FRESH run-dir: a stale partial would be read back as "already done" by the
# jsonl resume and would silently shrink the wave.
Remove-Item (Join-Path $k 'runs\ablation_1move_T6T4') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'ablation_1move_rc.txt') -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'ablation_1move_out.log') -Force -EA SilentlyContinue

try { schtasks.exe /delete /tn ablation_1move /f | Out-Null } catch {}
& schtasks.exe /create /tn ablation_1move /tr "$k\run_ablation_199.bat" /sc once /st 23:59 /f | Out-Null
$cc = $LASTEXITCODE
& schtasks.exe /run /tn ablation_1move | Out-Null
$rc = $LASTEXITCODE
Start-Sleep -Seconds 30
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'ablation_wave' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'ablation_1move_out.log'
Write-Output ("ABLATION armed: sha_ok=True create_rc=$cc run_rc=$rc python=$n master=$mc2" +
              "  log=$log  rc=$k\ablation_1move_rc.txt")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 40 }
