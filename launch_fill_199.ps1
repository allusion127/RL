# launch_fill_199.ps1 -- runs ON 199. Re-checks occupancy (non-intrusion) and
# REFUSES to launch if another user's MASTER is running, then starts fill_199.
$ErrorActionPreference='Continue'
$k='C:\Users\USER\lpopt_work\kit_frontier'
$m=@(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($m -gt 0) {
  Write-Output ("FILL199 REFUSED: master4.0m4_r1 count=$m (another user's job may be back) -- not launching")
  exit 1
}
try { schtasks.exe /delete /tn fill_199 /f | Out-Null } catch {}
& schtasks.exe /create /tn fill_199 /tr "$k\run_fill_199.bat" /sc once /st 23:59 /f | Out-Null
$cc=$LASTEXITCODE
& schtasks.exe /run /tn fill_199 | Out-Null
$rc=$LASTEXITCODE
Start-Sleep -Seconds 15
$py=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'fill_199' }).Count
$mc=@(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
Write-Output ('FILL199 create_rc='+$cc+' run_rc='+$rc+' python='+$py+' master='+$mc)
