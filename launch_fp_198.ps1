# launch_fp_198.ps1 -- runs ON 198. Registers + starts the fp_198 scheduled task
# (schtasks = the only launcher that survives ssh-session close). The task runs
# run_fp_198.bat (LPOPT_WORKER=1, flat_power LOW group E/J/K, workers 16 all-cores).
# USER runs this via: ssh USER@HOST_198 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fp_198.ps1"
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'
Remove-Item (Join-Path $k 'runs\flat_power_198') -Recurse -Force -ErrorAction SilentlyContinue
try { schtasks.exe /delete /tn fp_198 /f | Out-Null } catch {}
& schtasks.exe /create /tn fp_198 /tr "$k\run_fp_198.bat" /sc once /st 23:59 /f | Out-Null
$cc = $LASTEXITCODE
& schtasks.exe /run /tn fp_198 | Out-Null
$rc = $LASTEXITCODE
Start-Sleep -Seconds 15
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'flat_power_198' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fp_198_out.log'
Write-Output ('FP198 create_rc=' + $cc + ' run_rc=' + $rc + ' python=' + $py + ' master=' + $mc + ' log_bytes=' + $(if(Test-Path $log){(Get-Item $log).Length}else{-1}))
