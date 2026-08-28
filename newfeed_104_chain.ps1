# newfeed_104_chain.ps1 -- LOCAL (104) QUEUE stage 2: wait for the ga80 coverage-fill
# to finish, then run the NEW-FEED campaign (ga80 f105/f129/f137, 9 cells, P-cores).
# Safe to start now: polls until no fill_104 python remains.
$ErrorActionPreference = 'Continue'
$wd = $PSScriptRoot; Set-Location $wd
$env:LPOPT_WORKER='1'; $env:PYTHONUTF8='1'; $env:PYTHONUNBUFFERED='1'
"[queue2] waiting for fill_104 to finish $(Get-Date -Format 'HH:mm:ss')" |
    Out-File (Join-Path $wd 'newfeed_104_queue.log') -Encoding utf8
while ($true) {
  $n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
         Where-Object { $_.CommandLine -match 'fill_104\.inp' }).Count
  if ($n -eq 0) { break }
  Start-Sleep -Seconds 120
}
"[queue2] fill done -> new-feed campaign $(Get-Date -Format 'HH:mm:ss')" |
    Out-File (Join-Path $wd 'newfeed_104_queue.log') -Append -Encoding utf8
& python -u -m lpopt produce --input newfeed_104.inp *>&1 |
    Out-File (Join-Path $wd 'newfeed_104_out.log') -Encoding utf8
"DONE_$LASTEXITCODE" | Out-File (Join-Path $wd 'newfeed_104_rc.txt') -Encoding utf8
