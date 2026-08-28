# fp_104_launch.ps1 -- launch the LOCAL (104) flat_power worker as a durable
# scheduled task (survives session end). 104 is an AUTHORIZED worker box
# (explicit user permission 2026-07-25). LPOPT_WORKER=1 guard is set in the action.
# Fleet reshuffle 2026-07-25: 104 = P-CORES ONLY (deck use_all_cores=false pins
# workers 1:1 to P-cores; host -> E-cores). 104 works G3_G4 ONLY (highest-FE pair
# e_core 5.5), excludes E/J/K/L/N (20 cells). Seed 900, workers 8, round-budget 30,
# harvest_maps ON. (Disjoint: 199=E/J/K, 198=L/N, 104=G.)
# Korean path is passed ONLY via -WorkingDirectory (Unicode-safe); the cmd action
# string is 100% ASCII so nothing Korean passes through cmd parsing.
$ErrorActionPreference = 'Stop'
$wd = $PSScriptRoot
$py = (Get-Command python).Source
$excl = 'E1_E2_f113,E1_E2_f117,E1_E2_f121,E1_E2_f125,J1_J2_f113,J1_J2_f117,J1_J2_f121,J1_J2_f125,K1_K2_f113,K1_K2_f117,K1_K2_f121,K1_K2_f125,L1_L2_f113,L1_L2_f117,L1_L2_f121,L1_L2_f125,N1_N2_f113,N1_N2_f117,N1_N2_f121,N1_N2_f125'
$arg = '/c chcp 65001>nul & set LPOPT_WORKER=1&set PYTHONUTF8=1&set PYTHONUNBUFFERED=1&"' + $py + '" -u -m lpopt frontier-produce --input fp_104.inp --run-root runs/flat_power_104 --round-budget 30 --exclude-cells "' + $excl + '" > fp_104_out.log 2>&1&echo DONE_%ERRORLEVEL%>fp_104_rc.txt'
# clean any dry-run / prior artifacts so resume starts fresh
Remove-Item (Join-Path $wd 'runs\flat_power_104') -Recurse -Force -ErrorAction SilentlyContinue
try { Unregister-ScheduledTask -TaskName fp_104 -Confirm:$false -ErrorAction Stop } catch {}
$action    = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $arg -WorkingDirectory $wd
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName fp_104 -Action $action -Settings $settings -Principal $principal -Description 'flat_power local(104) worker: G3_G4 only, P-cores workers 8, seed 900' | Out-Null
Start-ScheduledTask -TaskName fp_104
Start-Sleep -Seconds 18
$info = Get-ScheduledTaskInfo -TaskName fp_104
$pyc  = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'flat_power_104' }).Count
$mc   = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log  = Join-Path $wd 'fp_104_out.log'
Write-Output ('FP104 launched: task_state=' + (Get-ScheduledTask -TaskName fp_104).State + ' last_result=' + $info.LastTaskResult + ' python_flat104=' + $pyc + ' master=' + $mc + ' log_bytes=' + $(if(Test-Path $log){(Get-Item $log).Length}else{-1}))
