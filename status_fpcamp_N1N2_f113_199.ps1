# status_fpcamp_N1N2_f113_199.ps1 -- READ-ONLY status probe for the N1_N2/f113
# low-feed opening campaign.  Emits a machine-parsable block; starts nothing.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_fpcamp_N1N2_f113_199.ps1"
$ErrorActionPreference = 'SilentlyContinue'
$k   = 'C:\Users\USER\lpopt_work\kit_frontier'
$r   = Join-Path $k 'runs\fpcamp_minfr_N1N2_f113'
$log = Join-Path $k 'fpcamp_minfr_N1N2_f113_out.log'
$rcf = Join-Path $k 'fpcamp_minfr_N1N2_f113_rc.txt'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'fpcamp_minfr_N1N2_f113' }).Count
Write-Output ("PROCS master=$mc python=$py")

if (Test-Path $rcf) { Write-Output ("RC " + ((Get-Content $rcf -Raw).Trim())) }

$st = Join-Path $r 'state.json'
if (Test-Path $st) {
  Write-Output '<<STATE'
  Get-Content $st -Raw
  Write-Output ''
  Write-Output 'STATE>>'
}

$lj = Join-Path $r 'labels.jsonl'
if (Test-Path $lj) { Write-Output ("NLABELS " + @(Get-Content $lj).Count) }

if (Test-Path $log) {
  Write-Output '<<WAVES'
  Select-String -Path $log -Pattern '^\[optimize\] wave' | Select-Object -Last 2 | ForEach-Object { $_.Line }
  Write-Output 'WAVES>>'
  Write-Output '<<FAILS'
  Select-String -Path $log -Pattern 'Traceback|CRITICAL|MissingCaseAsset|AssetResolutionError|ModelMismatch|MapHarvestAbort|ERROR|ABORT|HALT|halted' |
    Select-Object -Last 4 | ForEach-Object { $_.Line }
  Write-Output 'FAILS>>'
  Write-Output ("LOGBYTES " + (Get-Item $log).Length)
  Write-Output ("LOGMTIME " + (Get-Item $log).LastWriteTime.ToString('s'))
}
