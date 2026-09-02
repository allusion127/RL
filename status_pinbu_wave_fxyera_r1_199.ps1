# status_pinbu_wave_fxyera_r1_199.ps1 -- READ-ONLY status probe.  Starts nothing.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_pinbu_wave_fxyera_r1_199.ps1"
$ErrorActionPreference = 'SilentlyContinue'
$k   = 'C:\Users\USER\lpopt_work\kit_frontier'
$r   = Join-Path $k 'runs\pinbu_wave_fxyera_r1'
$log = Join-Path $k 'pinbu_wave_fxyera_r1_out.log'
$rcf = Join-Path $k 'pinbu_wave_fxyera_r1_rc.txt'
$res = Join-Path $r 'pinbu_wave_results.jsonl'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'pinbu_wave' }).Count
Write-Output ("PROCS master=$mc python=$py")

# RAM headroom: 12 concurrent MASTER chains is above the campaigns' proven 8, so
# the probe reports it -- a squeeze here is the one thing that would justify
# stopping and re-running at a lower worker count.  Disk is reported for the
# volume runs\ actually resolves to (it may be a junction to another drive).
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("FREERAM_GB " + [math]::Round($os.FreePhysicalMemory/1MB,1))
$runsDir = Join-Path $k 'runs'
$runsReal = (Get-Item $runsDir).Target
if (-not $runsReal) { $runsReal = $runsDir } else { $runsReal = @($runsReal)[0] }
$runsQual = Split-Path -Qualifier $runsReal
Write-Output ("RUNS_TARGET " + $runsReal)
Write-Output ("FREEDISK_GB " + [math]::Round((Get-PSDrive ($runsQual.Substring(0,1))).Free/1GB,1))

if (Test-Path $rcf) { Write-Output ("RC " + ((Get-Content $rcf -Raw).Trim())) }

# Retained final-cycle work dirs are the F_xy carrier (keep_success=true): the
# replay's FXYP is read from MAS_OUT OFFLINE -- it is not on WaveFom, so it never
# appears in the results JSONL.  Zero retained dirs with results present means
# keep_success did not take and the F_xy determinism arm is dead: report it, do
# not wait it out.
$wk = Join-Path $r 'paramA\master_work'
if (Test-Path $wk) {
  $dirs = @(Get-ChildItem $wk -Directory -Recurse -Depth 1 -EA SilentlyContinue |
            Where-Object { $_.Name -match '__MAS_RST' })
  $mo = @(Get-ChildItem $wk -Filter 'MAS_OUT*' -Recurse -EA SilentlyContinue)
  $ppi = @(Get-ChildItem $wk -Filter 'MAS_PPI*' -Recurse -EA SilentlyContinue)
  Write-Output ("RETAINED_CHAINDIRS " + $dirs.Count + "  MAS_OUT_FILES " + $mo.Count +
                "  MAS_PPI_FILES " + $ppi.Count)
} else {
  Write-Output 'RETAINED_CHAINDIRS 0  (master_work not created yet)'
}

if (Test-Path $res) {
  $objs = @(Get-Content $res | ForEach-Object { $_ | ConvertFrom-Json })
  Write-Output ("NRESULTS " + $objs.Count + " / 40")
  Write-Output '<<DONE'
  foreach ($o in $objs) {
    $pin = if ($null -eq $o.measured.max_pin_burnup) { '  --  ' } else { '{0,6:N3}' -f $o.measured.max_pin_burnup }
    $dfr = if ($null -eq $o.deltas_measured_minus_stored.f_r) { '   -   ' } else { '{0,7:N4}' -f $o.deltas_measured_minus_stored.f_r }
    Write-Output ("{0} {1,-17} {2,-12} pin={3} dFr={4} det={5} prov={6} {7}s" -f `
      $o.record_id.Substring(0,12), $o.group, $o.status, $pin, $dfr, `
      $o.determinism_ok, $o.provenance_ok, [math]::Round($o.wall_s,0))
  }
  Write-Output 'DONE>>'
  Write-Output ("CONVERGED " + @($objs | Where-Object { $_.status -eq 'converged' }).Count)
  Write-Output ("DETERMINISM_OK " + @($objs | Where-Object { $_.determinism_ok }).Count)
  Write-Output ("PROVENANCE_DRIFT " + @($objs | Where-Object { -not $_.provenance_ok }).Count)
  Write-Output ("PIN_PRESENT " + @($objs | Where-Object { $null -ne $_.measured.max_pin_burnup }).Count)
  Write-Output ("PIN_PASS_80 " + @($objs | Where-Object { $_.delivery_verdict -eq 'PASS' }).Count)
} else {
  Write-Output 'NRESULTS 0 / 40  (results file not created yet)'
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
