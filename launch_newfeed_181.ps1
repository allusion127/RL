# launch_newfeed_181.ps1 -- runs ON 181. NON-INTRUSION + integrity guards.
$ErrorActionPreference='Continue'
$k='C:\Users\USER\lpopt_work\kit_frontier'
# 1. non-intrusion: 181 is a SHARED box (SCALE jobs). Refuse if anyone is working.
$cpu=[math]::Round((Get-CimInstance Win32_Processor|Measure-Object -Property LoadPercentage -Average).Average,0)
$other=@(Get-Process -EA SilentlyContinue | Where-Object { $_.ProcessName -match 'scale|decart' }).Count
if ($other -gt 0) { Write-Output "NEWFEED181 REFUSED: other-user job detected (scale/decart procs=$other)"; exit 1 }
$busy=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'lpopt' }).Count
if ($busy -gt 0) { Write-Output "NEWFEED181 REFUSED: lpopt already running (python=$busy)"; exit 1 }
# 2. integrity: deck + code + champion must all be present (catch truncated ship)
foreach ($f in 'newfeed_181.inp','lpopt\data\edit5.py','data\models\20260725_063351\ensemble.json') {
  if (-not (Test-Path (Join-Path $k $f))) { Write-Output "NEWFEED181 REFUSED: missing $f"; exit 1 }
}
if (-not (Select-String -Path (Join-Path $k 'lpopt\data\edit5.py') -Pattern 'def stack_axial' -Quiet)) {
  Write-Output 'NEWFEED181 REFUSED: shipped code lacks the high-res harvest'; exit 1
}
try { schtasks.exe /delete /tn newfeed_181 /f | Out-Null } catch {}
& schtasks.exe /create /tn newfeed_181 /tr "$k\run_newfeed_181.bat" /sc once /st 23:59 /f | Out-Null
$cc=$LASTEXITCODE
& schtasks.exe /run /tn newfeed_181 | Out-Null
$rc=$LASTEXITCODE
Start-Sleep -Seconds 20
$n=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'newfeed_181' }).Count
Write-Output ("NEWFEED181 create_rc=$cc run_rc=$rc python=$n master=" + @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count + " cpu_before=$cpu%")
