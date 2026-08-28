$ErrorActionPreference='Continue'
$k='C:\Users\USER\lpopt_work\kit_frontier'
$other=@(Get-Process -EA SilentlyContinue | Where-Object { $_.ProcessName -match 'scale|decart' }).Count
if ($other -gt 0) { Write-Output "MAPCOV181 REFUSED: other-user job (scale/decart=$other)"; exit 1 }
$busy=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'lpopt' }).Count
$mc=@(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) { Write-Output "MAPCOV181 REFUSED: busy (python=$busy master=$mc)"; exit 1 }
foreach ($f in 'mapcov_181.inp','lpopt\data\flatness.py','data\models\20260725_063351\ensemble.json') {
  if (-not (Test-Path (Join-Path $k $f))) { Write-Output "MAPCOV181 REFUSED: missing $f"; exit 1 } }
try { schtasks.exe /delete /tn mapcov_181 /f | Out-Null } catch {}
& schtasks.exe /create /tn mapcov_181 /tr "$k\run_mapcov_181.bat" /sc once /st 23:59 /f | Out-Null
$cc=$LASTEXITCODE; & schtasks.exe /run /tn mapcov_181 | Out-Null; $rc=$LASTEXITCODE
Start-Sleep -Seconds 20
Write-Output ("MAPCOV181 create_rc=$cc run_rc=$rc python=" + @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'mapcov_181' }).Count + " master=" + @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count)
