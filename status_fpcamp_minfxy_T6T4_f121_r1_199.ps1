# status_fpcamp_minfxy_T6T4_f121_r1_199.ps1 -- READ-ONLY status probe for the
# programme's FIRST min_fxy campaign (T6_T4 / feed 121). Emits a machine-parsable
# block; starts nothing.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\status_fpcamp_minfxy_T6T4_f121_r1_199.ps1"
$ErrorActionPreference = 'SilentlyContinue'
$k   = 'C:\Users\USER\lpopt_work\kit_frontier'
$r   = Join-Path $k 'runs\fpcamp_minfxy_t6t4_f121_r1'
$log = Join-Path $k 'fpcamp_minfxy_t6t4_f121_r1_out.log'
$rcf = Join-Path $k 'fpcamp_minfxy_t6t4_f121_r1_rc.txt'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'fpcamp_minfxy_T6T4_f121_r1' }).Count
Write-Output ("PROCS master=$mc python=$py")

if (Test-Path $rcf) { Write-Output ("RC " + ((Get-Content $rcf -Raw).Trim())) }

$st = Join-Path $r 'state.json'
if (Test-Path $st) {
  Write-Output '<<STATE'
  Get-Content $st -Raw
  Write-Output ''
  Write-Output 'STATE>>'
}

$lj = Join-Path $r 'labels.jsonl'
if (Test-Path $lj) { Write-Output ("NLABELS " + @(Get-Content $lj).Count) }

# F_xy harvest health -- the ONE thing that can silently void this objective.
# A converged row with no f_xy is UNSCORABLE (-inf), not "worse".  NOTE each
# labels.jsonl line NESTS the store row under `.record` (verified against a
# stub dry-run of this exact deck); the gate fields live there, not at top level.
if (Test-Path $lj) {
  $rows = Get-Content $lj | ForEach-Object { ($_ | ConvertFrom-Json).record }
  $conv = @($rows | Where-Object { $_.converged })
  $lab  = @($conv | Where-Object { $null -ne $_.f_xy })
  Write-Output ("FXY_HARVEST converged=" + $conv.Count + " with_f_xy=" + $lab.Count)
  if ($conv.Count -gt 0 -and $lab.Count -lt $conv.Count) {
    Write-Output ("FXY_HARVEST WARNING: " + ($conv.Count - $lab.Count) + " converged row(s) carry NO f_xy -- they are UNSCORABLE on this objective, not merely worse")
  }
  if ($lab.Count -gt 0) {
    Write-Output '<<BESTFXY'
    $lab | Sort-Object f_xy | Select-Object -First 3 | ForEach-Object {
      Write-Output ("f_xy=" + $_.f_xy + " f_xya=" + $_.f_xya + " f_r=" + $_.f_r +
                    " cbc=" + $_.cbc_max + " f_q=" + $_.f_q + " ao=" + $_.ao_abs +
                    " cyclen=" + $_.cyclen + " pinbu=" + $_.max_pin_burnup +
                    " record=" + $_.record_id)
    }
    Write-Output 'BESTFXY>>'
  }
}

# state.json -> best_overall carries the FoM view: f_xy_margin_to_limit,
# deliverable, unknown_axes.  `deliverable` is EXPECTED false for every row this
# round (no PPI pin edit -> unknown_axes = ["max_pin_burnup"]); phase-2
# pinbu_wave supplies it.  `lpopt report`'s own table ranks by CYCLE DISTANCE
# regardless of objective and will NOT show the F_xy winner -- read this instead.
if (Test-Path $st) {
  $bo = (Get-Content $st -Raw | ConvertFrom-Json).best_overall
  if ($bo) {
    Write-Output ("BEST_OVERALL f_xy=" + $bo.f_xy + " margin=" + $bo.f_xy_margin_to_limit +
                  " limit=" + $bo.f_xy_limit_applied + " feasible=" + $bo.feasible +
                  " deliverable=" + $bo.deliverable +
                  " unknown_axes=[" + ($bo.unknown_axes -join ',') + "]")
  }
}

# fxy_source -- 'proxy' (s1i arm) or 'head' (s1j arm).  Decides what may be claimed.
$sel = Get-ChildItem -Path $r -Filter 'selection.json' -Recurse -EA SilentlyContinue
if ($sel) {
  Write-Output '<<FXYSOURCE'
  $sel | Select-Object -Last 2 | ForEach-Object {
    $s = (Get-Content $_.FullName -Raw | ConvertFrom-Json).fxy_source
    Write-Output ($_.Directory.Name + " fxy_source=" + $s)
  }
  Write-Output 'FXYSOURCE>>'
}

if (Test-Path $log) {
  Write-Output '<<BANNER'
  Select-String -Path $log -Pattern 'min_fxy objective|F_xy PROXY|predict_fxy' |
    Select-Object -First 3 | ForEach-Object { $_.Line }
  Write-Output 'BANNER>>'
  Write-Output '<<WAVES'
  Select-String -Path $log -Pattern '^\[optimize\] wave' | Select-Object -Last 3 | ForEach-Object { $_.Line }
  Write-Output 'WAVES>>'
  Write-Output '<<SEED'
  Select-String -Path $log -Pattern 'elite|store_elite|batches' | Select-Object -First 4 | ForEach-Object { $_.Line }
  Write-Output 'SEED>>'
  Write-Output '<<FAILS'
  Select-String -Path $log -Pattern 'Traceback|CRITICAL|MissingCaseAsset|AssetResolutionError|ModelMismatch|MapHarvestAbort|harvest_maps|ERROR|ABORT|HALT|halted|NOT DEFINED IN LPD|neutral' |
    Select-Object -Last 4 | ForEach-Object { $_.Line }
  Write-Output 'FAILS>>'
  Write-Output ("LOGBYTES " + (Get-Item $log).Length)
  Write-Output ("LOGMTIME " + (Get-Item $log).LastWriteTime.ToString('s'))
}
