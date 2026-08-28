# launch_fill_198.ps1 -- runs ON 198. Registers + starts the fill_198 QUEUE task.
# The task waits for the current flat_power round (fp_198_rc.txt) and then runs the
# coverage-fill produce campaign automatically -- safe to arm NOW, mid-flat_power.
# Coordinator runs:
#   ssh USER@HOST_198 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fill_198.ps1"
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'
try { schtasks.exe /delete /tn fill_198 /f | Out-Null } catch {}
& schtasks.exe /create /tn fill_198 /tr "$k\run_fill_198.bat" /sc once /st 23:59 /f | Out-Null
$cc = $LASTEXITCODE
& schtasks.exe /run /tn fill_198 | Out-Null
$rc = $LASTEXITCODE
Start-Sleep -Seconds 8
$q = Join-Path $k 'fill_198_queue.log'
$fpdone = Test-Path (Join-Path $k 'fp_198_rc.txt')
Write-Output ('FILL198 armed: create_rc=' + $cc + ' run_rc=' + $rc + ' queue_log=' + (Test-Path $q) + ' flat_power_done=' + $fpdone + ' (fill starts automatically when flat_power finishes)')
