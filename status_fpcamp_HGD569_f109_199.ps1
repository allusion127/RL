# status_fpcamp_HGD569_f109_199.ps1 -- READ-ONLY status probe for the high-Gd
# boron-opened-cell F_r assault.  Emits a machine-parsable block; starts nothing.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_fpcamp_HGD569_f109_199.ps1"
$ErrorActionPreference = 'SilentlyContinue'
$k   = 'C:\Users\USER\lpopt_work\kit_frontier'
$r   = Join-Path $k 'runs\fpcamp_minfr_hgd569_f109'
$log = Join-Path $k 'fpcamp_minfr_hgd569_f109_out.log'
$rcf = Join-Path $k 'fpcamp_minfr_hgd569_f109_rc.txt'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'fpcamp_minfr_HGD569_f109' }).Count
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
  Select-String -Path $log -Pattern '^\[optimize\] wave' | Select-Object -Last 3 | ForEach-Object { $_.Line }
  Write-Output 'WAVES>>'
  Write-Output '<<FAILS'
  Select-String -Path $log -Pattern 'Traceback|CRITICAL|MissingCaseAsset|AssetResolutionError|ModelMismatch|MapHarvestAbort|ERROR|ABORT|HALT|halted|NOT DEFINED IN LPD' |
    Select-Object -Last 4 | ForEach-Object { $_.Line }
  Write-Output 'FAILS>>'
  Write-Output ("LOGBYTES " + (Get-Item $log).Length)
  Write-Output ("LOGMTIME " + (Get-Item $log).LastWriteTime.ToString('s'))
}
