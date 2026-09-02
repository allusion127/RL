# status_intervention_wave_r1_199.ps1 -- READ-ONLY status probe for Campaign A
# round 1.  Emits a machine-parsable block; STARTS NOTHING and WRITES NOTHING.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_intervention_wave_r1_199.ps1"
#
# The two blocks that matter:
#   CELL   per-cell settled-chain counts off each cell's own ablation_results
#          .jsonl.  This wave is FIVE independent resumable sub-runs, so a
#          single global "n done" would hide a cell that died at chain 3.
#   FXY    per-cell fxy_sidecar.jsonl line counts.  The paired F_xy contrast is
#          this campaign's primary statistic; a healthy converged count with an
#          empty sidecar means the MAS_OUT retention chain broke and the wave is
#          producing F_r/cyclen only -- stop and diagnose before spending more
#          MASTER hours, exactly as the FXYERA1 MASOUT block does.
$ErrorActionPreference = 'SilentlyContinue'
$k   = 'C:\Users\USER\lpopt_work\kit_frontier'
$r   = Join-Path $k 'runs\intervention_wave_r1'
$log = Join-Path $k 'intervention_wave_r1_out.log'
$rcf = Join-Path $k 'intervention_wave_r1_rc.txt'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'intervention_wave' }).Count
Write-Output ("PROCS master=$mc python=$py")

$drive = (Get-Item $k).PSDrive.Name
Write-Output ("DISK ${drive}: freeGB=" + [math]::Round((Get-PSDrive $drive).Free / 1GB, 1))

if (Test-Path $rcf) { Write-Output ("RC " + ((Get-Content $rcf -Raw).Trim())) }

$cells = @('intervention_T6T4_f121', 'intervention_HGD569_f125',
           'intervention_E1E2_f121', 'intervention_E1E2_f109',
           'intervention_N1N2_f113')
$totDone = 0; $totConv = 0; $totFxy = 0
foreach ($c in $cells) {
  $d = Join-Path $r $c
  if (-not (Test-Path $d)) { Write-Output ("CELL $c n=0 conv=0 fxy=0 (no run dir)"); continue }
  $j = Join-Path $d 'ablation_results.jsonl'
  $s = Join-Path $d 'fxy_sidecar.jsonl'
  $n = 0; $conv = 0; $err = 0
  if (Test-Path $j) {
    $lines = @(Get-Content $j -EA SilentlyContinue)
    $n = $lines.Count
    $conv = @($lines | Where-Object { $_ -match '"status": ?"converged"' }).Count
    $err  = @($lines | Where-Object { $_ -match '"status": ?"error"' }).Count
  }
  $nf = 0
  if (Test-Path $s) { $nf = @(Get-Content $s -EA SilentlyContinue | Where-Object { $_ -match '"f_xy": ?[0-9]' }).Count }
  $kit = if (Test-Path (Join-Path $d 'kitdata\store')) { 'yes' } else { 'no' }
  Write-Output ("CELL $c n=$n conv=$conv err=$err fxy=$nf kit=$kit")
  $totDone += $n; $totConv += $conv; $totFxy += $nf
}
Write-Output ("TOTAL settled=$totDone converged=$totConv fxy_labels=$totFxy of 800 planned")

# -- retained MAS_OUT: the thing F_xy is parsed from -------------------------
if (Test-Path $r) {
  $outs = @(Get-ChildItem -Path $r -Recurse -Filter MAS_OUT -File -EA SilentlyContinue)
  $bytes = ($outs | Measure-Object -Property Length -Sum).Sum
  if ($null -eq $bytes) { $bytes = 0 }
  Write-Output ("MASOUT count=" + $outs.Count + " totalMB=" + [math]::Round($bytes / 1MB, 1))
  $newest = $outs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($newest) {
    $nfxy = @(Select-String -Path $newest.FullName -Pattern 'PLANAR POWER \(FXYP\)=' -EA SilentlyContinue).Count
    Write-Output ("FXYP newest=" + $newest.Name + " lines=" + $nfxy + " at=" + $newest.LastWriteTime.ToString('s'))
  }
  $rd = @(Get-ChildItem -Path $r -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum)
  Write-Output ("RUNDIR files=" + $rd.Count + " sizeGB=" + [math]::Round($rd.Sum / 1GB, 2))
} else {
  Write-Output "MASOUT count=0 (run dir does not exist yet)"
}

if (Test-Path $log) {
  Write-Output '<<CELLS'
  Select-String -Path $log -Pattern '^\[run\] ====|^\[run\] restart |^\[run\] kit ' | Select-Object -Last 12 | ForEach-Object { $_.Line }
  Write-Output 'CELLS>>'
  Write-Output '<<WAVES'
  Select-String -Path $log -Pattern '^\[run\] wave' | Select-Object -Last 6 | ForEach-Object { $_.Line }
  Write-Output 'WAVES>>'
  Write-Output '<<KIT'
  Select-String -Path $log -Pattern '^\[kit\]|record_id drift' | Select-Object -Last 6 | ForEach-Object { $_.Line }
  Write-Output 'KIT>>'
  Write-Output '<<FAILS'
  Select-String -Path $log -Pattern 'Traceback|CRITICAL|MissingCaseAsset|AssetResolutionError|unresolved|MapHarvestAbort|ERROR|ABORT|HALT|halted|REFUSED|restart drift|PER LIBRARY|No space left|WARNING' |
    Select-Object -Last 10 | ForEach-Object { $_.Line }
  Write-Output 'FAILS>>'
  Write-Output ("LOGBYTES " + (Get-Item $log).Length)
  Write-Output ("LOGMTIME " + (Get-Item $log).LastWriteTime.ToString('s'))
}
