# launch_pinbu_wave_fxyera_r1_199.ps1 -- runs ON 199.  Arms the PHASE-2 F_xy-era
# r1 wave (40 stored r1 cores replayed with enable_pin_burnup=True and
# keep_success=true: 20 F_xy-frontier pin-BU cores + 20 determinism replicates).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_pinbu_wave_fxyera_r1_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe.  schtasks no-ops silently on this fleet
# (data/reports/ab2_addendum_ADF_20260810.md section 3), so it is not used.
#
# BUSY GATE: REFUSES rather than stacks (v520_b2 precedent).  r1 must have drained
# (produce_fxyera_r1_prereg_20260829.md sec.6: "to be executed only after r1 drains").
#
# HASH GATES: deck, MANIFEST, harness and bat must be the artifacts the
# pre-registration hashed.  The manifest carries every pinned prediction and every
# stored label this wave is scored against -- a swapped manifest would silently
# un-register the experiment.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate ---------------------------------------------------------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|pinbu|produce' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("PINBU_FXYERA REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- hash gates --------------------------------------------------------------
$gates = @(
  @{ Path = Join-Path $k 'pinbu_wave_fxyera_r1_199.inp';
     Want = '9DF97D687652F7BA5FCD17234629AAF82134BED57E8A6CB790EB621E7620FD4B';
     Name = 'deck' },
  @{ Path = Join-Path $k 'data\reports\pinbu_wave_fxyera_r1_manifest.json';
     Want = '53D139C2BA7A42207EF0BED9DF26B558A62B76750AD9E88797434434E6E00D91';
     Name = 'manifest' },
  @{ Path = Join-Path $k 'pinbu_wave.py';
     Want = '5B3688CFAD684E9E837910F8842F68A2F2C21F931052DD52DE98262BA3581047';
     Name = 'harness' },
  @{ Path = Join-Path $k 'run_pinbu_wave_fxyera_r1_199.bat';
     Want = 'BD51ED2DF4762AF85DFCD88E4F340005E174DDB961B37D9D0FF1D25F28A2D2CC';
     Name = 'run bat' }
)
foreach ($g in $gates) {
  if (-not (Test-Path $g.Path)) {
    Write-Output ("PINBU_FXYERA REFUSED: " + $g.Name + " not found (" + $g.Path + ")"); exit 1
  }
  $got = (Get-FileHash -Algorithm SHA256 $g.Path).Hash
  if ($got -ne $g.Want) {
    Write-Output ("PINBU_FXYERA REFUSED: " + $g.Name + " sha256 mismatch")
    Write-Output ("  expected " + $g.Want)
    Write-Output ("  found    " + $got)
    exit 1
  }
}

# -- store gate: SHA256, not just length -------------------------------------
# The manifest pinned 40 record_ids and their stored f_xy / f_r / cyclen / cbc_max
# labels against THIS snapshot.  Those labels ARE the determinism reference; a
# different records.parquet means the reference moved and the wave is unscored.
# (The 2026-08-20 precedent gated on byte length only; sha256 costs the same here
# and is strictly stronger.)  Value = the coordinator's CURRENT local store,
# 2026-08-30, 75,793 rows.
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output 'PINBU_FXYERA REFUSED: store missing'; exit 1 }
$stLen = (Get-Item $st).Length
$stSha = (Get-FileHash -Algorithm SHA256 $st).Hash
$wantLen = 22538411
$wantSha = 'F38666E9F1508D35D33E0C22F583C5479C6F09CAC748201B494B47C8CFECA6EA'
if ($stLen -ne $wantLen -or $stSha -ne $wantSha) {
  Write-Output 'PINBU_FXYERA REFUSED: store is not the pinned snapshot'
  Write-Output ("  expected bytes=$wantLen sha=$wantSha")
  Write-Output ("  found    bytes=$stLen sha=$stSha")
  exit 1
}

# -- precondition gates ------------------------------------------------------
$ft = Join-Path $k 'data\store\fuel_types.parquet'
if (-not (Test-Path $ft)) { Write-Output 'PINBU_FXYERA REFUSED: fuel_types.parquet missing'; exit 1 }
# ALL 40 targets are paramA -- the ga80 FEASIBLE_PACKAGE is never resolved by this
# wave, so its absence is reported, not refused.
foreach ($p in @('data\design\package\cores', 'data\design\package\lib')) {
  if (-not (Test-Path (Join-Path $k $p))) {
    Write-Output "PINBU_FXYERA REFUSED: $p missing (paramA routing)"; exit 1
  }
}
if (-not (Test-Path (Join-Path $k 'FEASIBLE_PACKAGE'))) {
  Write-Output 'PINBU_FXYERA NOTE: FEASIBLE_PACKAGE absent -- fine, no ga80 target in this wave'
}
# PROVENANCE CACHE.  18 of the 40 targets carry restart_provenance "promoted:*",
# which resolves out of the produce promoted cache.  If that cache has been pruned
# the replay silently falls back to pair_ecore/pair_feed, provenance_ok goes False
# and "pinbu_wave.py patch" refuses the pin value -- the exact failure that dropped
# 3 of 44 chains on 2026-08-20.  Refuse up front instead of burning 1.6 h.
$prom = Join-Path $k 'data\produce\promoted'
if (-not (Test-Path $prom)) {
  Write-Output 'PINBU_FXYERA REFUSED: data\produce\promoted missing -- 18/40 targets need it'; exit 1
}
$nprom = @(Get-ChildItem $prom -Directory -EA SilentlyContinue).Count
Write-Output ("PINBU_FXYERA promoted cache cells: $nprom")
if (-not (Test-Path 'C:/DeCART_MASTER/BIN/master4.0m4_r1.exe')) {
  Write-Output 'PINBU_FXYERA REFUSED: MASTER executable not found'; exit 1
}
# 12 concurrent chains: refuse on a box that cannot hold them.  Disk is checked on
# the volume runs\ ACTUALLY resolves to -- runs\ may be a junction to another
# drive, in which case C: free space is the wrong number.
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB,1)
$runsDir = Join-Path $k 'runs'
if (-not (Test-Path $runsDir)) { New-Item -ItemType Directory -Path $runsDir | Out-Null }
$runsReal = (Get-Item $runsDir).Target
if (-not $runsReal) { $runsReal = $runsDir } else { $runsReal = @($runsReal)[0] }
$runsQual = Split-Path -Qualifier $runsReal
$diskGB = [math]::Round((Get-PSDrive ($runsQual.Substring(0,1))).Free/1GB,1)
Write-Output ("PINBU_FXYERA runs -> $runsReal (volume $runsQual free ${diskGB}GB)")
if ($freeGB -lt 30) { Write-Output "PINBU_FXYERA REFUSED: only $freeGB GB RAM free (need >= 30)"; exit 1 }
if ($diskGB -lt 15) { Write-Output "PINBU_FXYERA REFUSED: only $diskGB GB free on the runs volume (need >= 15)"; exit 1 }

# -- FRESH run dir -----------------------------------------------------------
# The harness RESUMES from pinbu_wave_results.jsonl, so a stale partial would be
# read back as "already measured" and silently shrink the wave.  Removing it is
# safe here because nothing has been merged yet; after a real partial run, do NOT
# re-run this launcher -- re-invoke the bat, which resumes.
# Only the wave's OWN subdir is removed; runs\ itself (possibly a junction) is not.
Remove-Item (Join-Path $k 'runs\pinbu_wave_fxyera_r1') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_fxyera_r1_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_fxyera_r1_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$bat = Join-Path $k 'run_pinbu_wave_fxyera_r1_199.bat'
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("PINBU_FXYERA Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output 'PINBU_FXYERA LAUNCH FAILED (non-zero ReturnValue)'; exit 1 }

Start-Sleep -Seconds 60
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'pinbu_wave' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'pinbu_wave_fxyera_r1_out.log'
Write-Output ("PINBU_FXYERA armed: hashes_ok=True python=$n master=$mc2 freeRAM=${freeGB}GB" +
              "  log=$log  rc=$k\pinbu_wave_fxyera_r1_rc.txt")
if (Test-Path $log) {
  Write-Output '--- log header ---'
  Get-Content $log -TotalCount 40
} else {
  Write-Output '--- log not created yet ---'
}
