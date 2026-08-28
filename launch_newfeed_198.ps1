# launch_newfeed_198.ps1 -- arm QUEUE stage 2 on 198 (paramA new feeds).
# Chains AFTER fill_198. Safe to arm now, alongside the fill queue.
$ErrorActionPreference='Continue'
$k='C:\Users\USER\lpopt_work\kit_frontier'
try { schtasks.exe /delete /tn newfeed_198 /f | Out-Null } catch {}
& schtasks.exe /create /tn newfeed_198 /tr "$k\run_newfeed_198.bat" /sc once /st 23:59 /f | Out-Null
$cc=$LASTEXITCODE
& schtasks.exe /run /tn newfeed_198 | Out-Null
$rc=$LASTEXITCODE
Start-Sleep -Seconds 6
Write-Output ('NEWFEED198 armed: create_rc='+$cc+' run_rc='+$rc+' deck='+(Test-Path (Join-Path $k 'newfeed_198.inp'))+' (starts when fill_198 completes)')
