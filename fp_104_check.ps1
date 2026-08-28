# fp_104_check.ps1 -- LIGHTWEIGHT local monitor for the 104 flat_power worker.
# Read-only: filesystem + Get-Process + one CPU counter. NO python spawn (avoids the
# 199-style monitor-overload). Prints one status line. Loop it from outside with a
# sleep if desired (e.g.  while($true){ .\fp_104_check.ps1; Start-Sleep 120 } ).
$wd  = $PSScriptRoot
$rr  = Join-Path $wd 'runs\flat_power_104'
$conv = 0; $maps = '-'; $firstcell = '-'
if (Test-Path $rr) {
    $lbls = Get-ChildItem $rr -Recurse -Filter labels.jsonl -EA SilentlyContinue
    foreach ($f in $lbls) {
        $lines = @(Get-Content $f.FullName -EA SilentlyContinue | Where-Object { $_ -match '"converged"\s*:\s*true' })
        $conv += $lines.Count
        if ($conv -gt 0 -and $firstcell -eq '-') { $firstcell = Split-Path (Split-Path $f.FullName -Parent) -Leaf }
        if ($conv -gt 0 -and $maps -eq '-') { if ($lines[0] -match '"maps_key"\s*:\s*"([^"]+)"') { $maps = 'SET' } }
    }
}
$rs  = Join-Path $rr 'race_state.json'
$excl = '-'
if (Test-Path $rs) { try { $j = Get-Content $rs -Raw | ConvertFrom-Json; if ($j.exclusions) { $excl = (@($j.exclusions.PSObject.Properties.Name)).Count } } catch {} }
$pyc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'flat_power_104' }).Count
$mc  = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$cpu = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average,0)
$rc  = Join-Path $wd 'fp_104_rc.txt'
$rcv = if (Test-Path $rc) { (Get-Content $rc -Raw).Trim() } else { '-' }
Write-Output ('[fp104 ' + (Get-Date -Format HH:mm) + '] py=' + $pyc + ' master=' + $mc + ' cpu=' + $cpu + '% converged=' + $conv + ' first_cell=' + $firstcell + ' maps_key=' + $maps + ' excluded_cells=' + $excl + ' rc=' + $rcv)
