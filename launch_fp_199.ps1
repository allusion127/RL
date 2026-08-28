# launch_fp_199.ps1 -- runs ON 199 (after starvation cleared + deck/bat shipped).
# Registers + starts the fp_199 scheduled task running run_fp_199.bat
# (LPOPT_WORKER=1, flat_power LOW group E/J/K, workers 24 all-cores).
# USER runs this via: ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fp_199.ps1"
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'
Remove-Item (Join-Path $k 'runs\flat_power_199') -Recurse -Force -ErrorAction SilentlyContinue
try { schtasks.exe /delete /tn fp_199 /f | Out-Null } catch {}
& schtasks.exe /create /tn fp_199 /tr "$k\run_fp_199.bat" /sc once /st 23:59 /f | Out-Null
$cc = $LASTEXITCODE
& schtasks.exe /run /tn fp_199 | Out-Null
$rc = $LASTEXITCODE
Start-Sleep -Seconds 15
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'flat_power_199' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fp_199_out.log'
Write-Output ('FP199 create_rc=' + $cc + ' run_rc=' + $rc + ' python=' + $py + ' master=' + $mc + ' log_bytes=' + $(if(Test-Path $log){(Get-Item $log).Length}else{-1}))
