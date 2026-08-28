# launch_fill_198b.ps1 -- 198 STAGE 3. Refuses to start if a job is already running
# (rc==0 gate discipline: never stack campaigns on one box again).
$ErrorActionPreference='Continue'
$k='C:\Users\USER\lpopt_work\kit_frontier'
$busy=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'lpopt' }).Count
$mc=@(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("FILL198B REFUSED: box busy (lpopt_python=$busy master=$mc)"); exit 1
}
# model/schema consistency: the deck must point at a champion that EXISTS here
if (-not (Test-Path (Join-Path $k 'data\models\20260725_063351\ensemble.json'))) {
  Write-Output 'FILL198B REFUSED: A6 champion (20260725_063351) not found on this box'; exit 1
}
try { schtasks.exe /delete /tn fill_198b /f | Out-Null } catch {}
& schtasks.exe /create /tn fill_198b /tr "$k\run_fill_198b.bat" /sc once /st 23:59 /f | Out-Null
$cc=$LASTEXITCODE
& schtasks.exe /run /tn fill_198b | Out-Null
$rc=$LASTEXITCODE
Start-Sleep -Seconds 20
$n=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
     Where-Object { $_.CommandLine -match 'fill_198b' }).Count
Write-Output ("FILL198B create_rc=$cc run_rc=$rc python=$n master=" + @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count)
