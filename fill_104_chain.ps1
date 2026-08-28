# fill_104_chain.ps1 -- LOCAL (104) QUEUE: wait for the running flat_power round
# to exit, then start the ga80 coverage-fill on P-CORES ONLY.
# Safe to start NOW (mid-flat_power): it polls until no flat_power_104 python
# remains, then runs `lpopt produce --input fill_104.inp`.
# Coordinator runs (background, same way fp_104 was launched):
#   powershell -NoProfile -ExecutionPolicy Bypass -File "<5_RL>\fill_104_chain.ps1"
$ErrorActionPreference = 'Continue'
$wd = $PSScriptRoot
Set-Location $wd
$env:LPOPT_WORKER = '1'
$env:PYTHONUTF8   = '1'
$env:PYTHONUNBUFFERED = '1'
"[queue] waiting for flat_power_104 to finish  $(Get-Date -Format 'HH:mm:ss')" |
    Out-File -FilePath (Join-Path $wd 'fill_104_queue.log') -Encoding utf8
while ($true) {
    $n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
           Where-Object { $_.CommandLine -match 'flat_power_104' }).Count
    if ($n -eq 0) { break }
    Start-Sleep -Seconds 120
}
"[queue] flat_power done -> starting coverage-fill  $(Get-Date -Format 'HH:mm:ss')" |
    Out-File -FilePath (Join-Path $wd 'fill_104_queue.log') -Append -Encoding utf8
& python -u -m lpopt produce --input fill_104.inp *>&1 |
    Out-File -FilePath (Join-Path $wd 'fill_104_out.log') -Encoding utf8
"DONE_$LASTEXITCODE" | Out-File -FilePath (Join-Path $wd 'fill_104_rc.txt') -Encoding utf8
