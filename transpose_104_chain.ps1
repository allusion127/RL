# transpose_104_chain.ps1 -- LOCAL (104): wait for the coverage-fill to finish,
# then run the transpose-pair reproducibility experiment (24 pairs = 48 calls).
# Measures the MASTER label noise floor DIRECTLY (same physics, different encoding).
$ErrorActionPreference='Continue'
$wd=$PSScriptRoot; Set-Location $wd
$env:LPOPT_WORKER='1'; $env:PYTHONUTF8='1'; $env:PYTHONUNBUFFERED='1'
"[tp] waiting for fill_104 $(Get-Date -Format 'HH:mm:ss')" | Out-File (Join-Path $wd 'transpose_104_queue.log') -Encoding utf8
while ($true) {
  $n=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fill_104\.inp' }).Count
  if ($n -eq 0) { break }
  Start-Sleep -Seconds 120
}
"[tp] fill done -> transpose pairs $(Get-Date -Format 'HH:mm:ss')" | Out-File (Join-Path $wd 'transpose_104_queue.log') -Append -Encoding utf8
& python -u transpose_pairs.py --deck fill_104.inp --n 24 *>&1 | Out-File (Join-Path $wd 'transpose_104_out.log') -Encoding utf8
"DONE_$LASTEXITCODE" | Out-File (Join-Path $wd 'transpose_104_rc.txt') -Encoding utf8
