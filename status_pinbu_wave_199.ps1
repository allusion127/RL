# status_pinbu_wave_199.ps1 -- READ-ONLY status probe.  Starts nothing.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_pinbu_wave_199.ps1"
$ErrorActionPreference = 'SilentlyContinue'
$k   = 'C:\Users\USER\lpopt_work\kit_frontier'
$r   = Join-Path $k 'runs\pinbu_wave'
$log = Join-Path $k 'pinbu_wave_out.log'
$rcf = Join-Path $k 'pinbu_wave_rc.txt'
$res = Join-Path $r 'pinbu_wave_results.jsonl'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'pinbu_wave' }).Count
Write-Output ("PROCS master=$mc python=$py")

# RAM headroom: 12 concurrent MASTER chains is above the campaigns' proven 8, so
# the probe reports it -- a squeeze here is the one thing that would justify
# stopping and re-running at a lower worker count.
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("FREERAM_GB " + [math]::Round($os.FreePhysicalMemory/1MB,1))
Write-Output ("FREEDISK_GB " + [math]::Round((Get-PSDrive C).Free/1GB,1))

if (Test-Path $rcf) { Write-Output ("RC " + ((Get-Content $rcf -Raw).Trim())) }

if (Test-Path $res) {
  $lines = @(Get-Content $res)
  Write-Output ("NRESULTS " + $lines.Count + " / 44")
  Write-Output '<<DONE'
  foreach ($l in $lines) {
    $o = $l | ConvertFrom-Json
    $pin = if ($null -eq $o.measured.max_pin_burnup) { '  --  ' } else { '{0,6:N3}' -f $o.measured.max_pin_burnup }
    Write-Output ("{0} {1,-20} {2,-12} pin={3} det={4} prov={5} {6}s" -f `
      $o.record_id.Substring(0,12), $o.group, $o.status, $pin, `
      $o.determinism_ok, $o.provenance_ok, [math]::Round($o.wall_s,0))
  }
  Write-Output 'DONE>>'
} else {
  Write-Output 'NRESULTS 0 / 44  (results file not created yet)'
}

if (Test-Path $log) {
  Write-Output '<<TAIL'
  Get-Content $log -Tail 12
  Write-Output 'TAIL>>'
  Write-Output '<<FAILS'
  Select-String -Path $log -Pattern 'Traceback|CRITICAL|MissingCaseAsset|AssetResolutionError|record_id drift|SKIP|ERROR|ABORT|HALT|NOT DEFINED IN LPD|non_finite' |
    Select-Object -Last 6 | ForEach-Object { $_.Line }
  Write-Output 'FAILS>>'
  Write-Output ("LOGBYTES " + (Get-Item $log).Length)
  Write-Output ("LOGMTIME " + (Get-Item $log).LastWriteTime.ToString('s'))
}
