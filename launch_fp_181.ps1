# launch_fp_181.ps1 -- runs ON 181 (USER@HOST_181, password auth via pm181.py).
# Registers + starts the fp_181 scheduled task running run_fp_181.bat
# (LPOPT_WORKER=1, flat_power L1_L2+N1_N2, workers 16 all-cores). Coordinator runs:
#   python pm181.py exec "& 'C:/Users/USER/lpopt_work/kit_frontier/launch_fp_181.ps1'"
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'
Remove-Item (Join-Path $k 'runs\flat_power_181') -Recurse -Force -ErrorAction SilentlyContinue
try { schtasks.exe /delete /tn fp_181 /f | Out-Null } catch {}
& schtasks.exe /create /tn fp_181 /tr "$k\run_fp_181.bat" /sc once /st 23:59 /f | Out-Null
$cc = $LASTEXITCODE
& schtasks.exe /run /tn fp_181 | Out-Null
$rc = $LASTEXITCODE
Start-Sleep -Seconds 15
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'flat_power_181' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fp_181_out.log'
Write-Output ('FP181 create_rc=' + $cc + ' run_rc=' + $rc + ' python=' + $py + ' master=' + $mc + ' log_bytes=' + $(if(Test-Path $log){(Get-Item $log).Length}else{-1}))
