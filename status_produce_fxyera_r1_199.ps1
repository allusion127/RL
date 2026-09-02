# status_produce_fxyera_r1_199.ps1 -- READ-ONLY status probe for the F_xy-era
# coverage-fill produce round 1.  Emits a machine-parsable block; STARTS NOTHING
# and WRITES NOTHING.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_produce_fxyera_r1_199.ps1"
#
# The MASOUT block is the one that matters: retained MAS_OUT is this round's
# actual deliverable (F_xy is printed nowhere else), so a healthy wave count with
# MASOUT 0 means the retention chain broke and the round is producing scalars
# only -- stop and diagnose before spending more MASTER hours.
$ErrorActionPreference = 'SilentlyContinue'
$k   = 'C:\Users\USER\lpopt_work\kit_frontier'
$r   = Join-Path $k 'runs\produce_fxyera_r1'
$log = Join-Path $k 'produce_fxyera_r1_out.log'
$rcf = Join-Path $k 'produce_fxyera_r1_rc.txt'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'produce_fxyera_r1' }).Count
Write-Output ("PROCS master=$mc python=$py")

$drive = (Get-Item $k).PSDrive.Name
Write-Output ("DISK ${drive}: freeGB=" + [math]::Round((Get-PSDrive $drive).Free / 1GB, 1))

if (Test-Path $rcf) { Write-Output ("RC " + ((Get-Content $rcf -Raw).Trim())) }

# -- the deliverable: retained final-cycle dirs carrying MAS_OUT --------------
if (Test-Path $r) {
  $outs = @(Get-ChildItem -Path $r -Recurse -Filter MAS_OUT -File -EA SilentlyContinue)
  $bytes = ($outs | Measure-Object -Property Length -Sum).Sum
  if ($null -eq $bytes) { $bytes = 0 }
  Write-Output ("MASOUT count=" + $outs.Count + " totalMB=" + [math]::Round($bytes / 1MB, 1))
  $rd = @(Get-ChildItem -Path $r -Recurse -File -EA SilentlyContinue |
          Measure-Object -Property Length -Sum)
  Write-Output ("RUNDIR files=" + $rd.Count + " sizeGB=" + [math]::Round($rd.Sum / 1GB, 2))
  # Spot-check that the newest retained MAS_OUT really carries the FXYP line the
  # F_xy parser will read.  One file, tail only -- cheap and decisive.
  $newest = $outs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($newest) {
    $nfxy = @(Select-String -Path $newest.FullName -Pattern 'PLANAR POWER \(FXYP\)=' -EA SilentlyContinue).Count
    Write-Output ("FXYP newest=" + $newest.Name + " lines=" + $nfxy + " at=" + $newest.LastWriteTime.ToString('s'))
  }
} else {
  Write-Output "MASOUT count=0 (run dir does not exist yet)"
}

if (Test-Path $log) {
  Write-Output '<<WAVES'
  Select-String -Path $log -Pattern '^\[produce\] wave' | Select-Object -Last 6 | ForEach-Object { $_.Line }
  Write-Output 'WAVES>>'
  Write-Output '<<RESUME'
  Select-String -Path $log -Pattern 'resumed|workers|promoted' | Select-Object -First 6 | ForEach-Object { $_.Line }
  Write-Output 'RESUME>>'
  Write-Output '<<SUMMARY'
  Select-String -Path $log -Pattern 'produce summary|waves=|RESULT:|elite_perturb degraded' | Select-Object -Last 6 | ForEach-Object { $_.Line }
  Write-Output 'SUMMARY>>'
  Write-Output '<<FAILS'
  Select-String -Path $log -Pattern 'Traceback|CRITICAL|MissingCaseAsset|AssetResolutionError|unresolved|MapHarvestAbort|ERROR|ABORT|HALT|halted|NOT DEFINED IN LPD|neutral|STALLED|WARNING' |
    Select-Object -Last 8 | ForEach-Object { $_.Line }
  Write-Output 'FAILS>>'
  Write-Output ("LOGBYTES " + (Get-Item $log).Length)
  Write-Output ("LOGMTIME " + (Get-Item $log).LastWriteTime.ToString('s'))
}
