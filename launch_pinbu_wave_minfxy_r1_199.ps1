# launch_pinbu_wave_minfxy_r1_199.ps1 -- runs ON 199.  Arms the PHASE-2 min_fxy
# T6_T4/f121 r1 wave (25 stored paramA cores replayed with enable_pin_burnup=True
# and keep_success=true: 20 F_xy-frontier pin-BU cores of
# fpcamp_minfxy_t6t4_f121_r1 + 5 F_r-era cores measured for F_xy for the first time).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_pinbu_wave_minfxy_r1_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe.  schtasks no-ops silently on this fleet
# (data/reports/ab2_addendum_ADF_20260810.md section 3), so it is not used.
#
# BUSY GATE: REFUSES rather than stacks (v520_b2 precedent).  This box is shared
# with intervention_wave_r1 (Campaign A round 1, 800 chains, ~14.4 h at 24
# workers), which owns the whole MASTER queue while it runs.  Two MASTER queues
# on one box is how a wave silently loses its cadence -- and this wave's own
# budget claim is calibrated on an EMPTY box.  The gate names
# `intervention_wave` explicitly: the fxyera launcher's regex did NOT match it.
#
# PROMOTED-CACHE CELL-COUNT GATE: the recommended next step of
# pinbu_wave_fxyera_r1_results_20260830.md section 8.  See "$wantPromCells" below --
# it is FAIL-CLOSED and must be stamped from the status probe before arming.
#
# HASH GATES: deck, MANIFEST, harness and bat must be the artifacts the
# pre-registration hashed.  The manifest carries every pinned prediction and every
# stored label this wave is scored against -- a swapped manifest would silently
# un-register the experiment.
#
# Nothing here touches runs\intervention_wave_r1, runs\pinbu_wave_fxyera_r1,
# runs\fpcamp_minfxy_t6t4_f121_r1, any other runs\ dir, or boxes 198 / 181 / 238.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

function Refuse($msg) { Write-Output ("PINBU_MINFXY REFUSED: " + $msg); exit 1 }

# -- busy gate ---------------------------------------------------------------
# `intervention` and `mesh` added to the fxyera regex: intervention_wave_r1 runs
# as `python.exe -u intervention_wave.py run ...` and would NOT have matched
# 'lpopt|ablation|batchswap|pinbu|produce' (it imports ablation_wave, it does not
# run it).  A resume of its ga80 leg
# (resume_intervention_wave_r1_ga80_199.bat) has the same command line.
$busyProcs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
               Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|pinbu|produce|intervention|mesh' })
$busy = $busyProcs.Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("PINBU_MINFXY REFUSED: box busy (python=$busy master=$mc)")
  foreach ($p in $busyProcs) {
    Write-Output ("  busy pid=" + $p.ProcessId + " : " + $p.CommandLine.Substring(0, [Math]::Min(160, $p.CommandLine.Length)))
  }
  exit 1
}

# -- hash gates --------------------------------------------------------------
$gates = @(
  @{ Path = Join-Path $k 'pinbu_wave_minfxy_r1_199.inp';
     Want = '1C27E3971DE6010E95477A1451FD380E8DEE2BAD992D177F0517DF19030E6C88';
     Name = 'deck' },
  @{ Path = Join-Path $k 'data\reports\pinbu_wave_minfxy_r1_manifest.json';
     Want = 'BC035D442891A79A640E0BFD7A3FAAB58B24A1D5E7D5A7F65494B268A3C35041';
     Name = 'manifest' },
  @{ Path = Join-Path $k 'pinbu_wave.py';
     Want = '5B3688CFAD684E9E837910F8842F68A2F2C21F931052DD52DE98262BA3581047';
     Name = 'harness' },
  @{ Path = Join-Path $k 'run_pinbu_wave_minfxy_r1_199.bat';
     Want = 'A23502217AFB397A3C65B9822BFDA3CAADA8CBA14999D301FA89C0A45AFDDB3F';
     Name = 'run bat' }
)
foreach ($g in $gates) {
  if (-not (Test-Path $g.Path)) { Refuse ($g.Name + " not found (" + $g.Path + ")") }
  $got = (Get-FileHash -Algorithm SHA256 $g.Path).Hash
  if ($got -ne $g.Want) {
    Write-Output ("PINBU_MINFXY REFUSED: " + $g.Name + " sha256 mismatch")
    Write-Output ("  expected " + $g.Want)
    Write-Output ("  found    " + $got)
    exit 1
  }
}

# -- store gate: SHA256, not just length -------------------------------------
# The manifest pinned 25 record_ids and their stored f_xy / f_r / cyclen / cbc_max
# labels against THIS snapshot.  Those labels ARE the determinism reference; a
# different records.parquet means the reference moved and the wave is unscored.
# Value = the coordinator's CURRENT local store, 2026-08-30, 75,893 rows
# (75,793 of the F_xy-era merge + the 100 rows of fpcamp_minfxy_t6t4_f121_r1).
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Refuse 'store missing' }
$stLen = (Get-Item $st).Length
$stSha = (Get-FileHash -Algorithm SHA256 $st).Hash
$wantLen = 22780281  # re-stamped 2026-08-30 with the intervention-merged store
$wantSha = '73701E33F07291E17609BA30D025E2A5B7A423FEB69F08D23DE4EC23EBE0C85F'  # re-stamped 2026-08-30 after intervention_wave_r1 merge (76,693 rows)
if ($stLen -ne $wantLen -or $stSha -ne $wantSha) {
  Write-Output 'PINBU_MINFXY REFUSED: store is not the pinned snapshot'
  Write-Output ("  expected bytes=$wantLen sha=$wantSha")
  Write-Output ("  found    bytes=$stLen sha=$stSha")
  exit 1
}

# -- precondition gates ------------------------------------------------------
$ft = Join-Path $k 'data\store\fuel_types.parquet'
if (-not (Test-Path $ft)) { Refuse 'fuel_types.parquet missing' }
# ALL 25 targets are paramA -- the ga80 FEASIBLE_PACKAGE is never resolved by this
# wave, so its absence is reported, not refused.
foreach ($p in @('data\design\package\cores', 'data\design\package\lib')) {
  if (-not (Test-Path (Join-Path $k $p))) { Refuse "$p missing (paramA routing)" }
}
if (-not (Test-Path (Join-Path $k 'FEASIBLE_PACKAGE'))) {
  Write-Output 'PINBU_MINFXY NOTE: FEASIBLE_PACKAGE absent -- fine, no ga80 target in this wave'
}
if (-not (Test-Path 'C:/DeCART_MASTER/BIN/master4.0m4_r1.exe')) {
  Refuse 'MASTER executable not found'
}

# -- PROVENANCE GATE, read from the MANIFEST ---------------------------------
# The fix for the failure that refused 8/40 on pinbu_wave_fxyera_r1 (results
# 20260830 section 1) is upstream of this launcher: provenance was resolved at
# PLAN time, and only rows whose stored restart_provenance is `native:` were
# selected.  All 25 share one restart, native:MAS_RST.APRQ_10_0615.11, which
# resolves out of data\design\package and NOT out of data\produce\promoted -- so
# no promoted-cache state, grown or pruned, can move a resolution.
# This gate re-asserts that property against the manifest actually on disk, so a
# future edited manifest cannot quietly reintroduce a promoted-dependent target.
$manPath = Join-Path $k 'data\reports\pinbu_wave_minfxy_r1_manifest.json'
$man = Get-Content $manPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($man.n_targets -ne 25) { Refuse ("manifest declares n_targets=" + $man.n_targets + ", expected 25") }
$nonNative = @($man.targets | Where-Object { $_.restart_provenance -notlike 'native:*' })
if ($nonNative.Count -gt 0) {
  Write-Output ("PINBU_MINFXY REFUSED: " + $nonNative.Count +
                " target(s) are not native: provenance -- the plan-time provenance rule was violated")
  foreach ($t in $nonNative) {
    Write-Output ("  " + $t.record_id.Substring(0,12) + "  " + $t.restart_provenance)
  }
  exit 1
}
$restarts = @($man.targets | ForEach-Object { $_.restart_provenance } | Sort-Object -Unique)
Write-Output ("PINBU_MINFXY provenance: 25/25 native, restart file(s): " + ($restarts -join ', '))

# -- PROMOTED-CACHE CELL-COUNT GATE ------------------------------------------
# pinbu_wave_fxyera_r1_results_20260830.md section 8, first recommended next step:
# "A launcher gate on promoted-cache CELL COUNT (not presence), which would have
# caught section 1's drift before 0.58 h was spent."  There, the fxyera launcher
# only checked that data\produce\promoted EXISTED; the cache had GROWN, and the
# growth is what re-resolved 8 pair_* targets onto promoted restarts.
#
# This wave has ZERO promoted-dependent targets, so the count is a CHANGE
# DETECTOR on the resolution environment, not a dependency.  It is still gated,
# because a cache that moved between stamping and arming means the box's produce
# state moved, and that is worth refusing 25 chains over.
#
# FAIL-CLOSED.  -1 is the unstamped sentinel: the launcher REFUSES until the
# operator has read the count off the read-only status probe (which prints
# PROMOTED_CELLS) and stamped it here, then re-hashed this file into the prereg.
# This is prereg section 9, step 0.  The coordinator cannot observe 199's cache,
# so the value is deliberately NOT guessed.
$wantPromCells = 8  # stamped 2026-08-30 17:xx from HOST_199 dataproducepromoted (step 0)
$prom = Join-Path $k 'data\produce\promoted'
if (-not (Test-Path $prom)) { Refuse 'data\produce\promoted missing' }
$nprom = @(Get-ChildItem $prom -Directory -EA SilentlyContinue).Count
Write-Output ("PINBU_MINFXY promoted cache cells: $nprom")
if ($wantPromCells -lt 0) {
  Write-Output 'PINBU_MINFXY REFUSED: promoted-cache cell-count gate is UNSTAMPED.'
  Write-Output ("  observed cell count on this box: " + $nprom)
  Write-Output '  Set $wantPromCells to that number in this file, re-hash it into'
  Write-Output '  data/reports/pinbu_wave_minfxy_r1_prereg_20260830.md section 8, then re-arm.'
  exit 1
}
if ($nprom -ne $wantPromCells) {
  Write-Output 'PINBU_MINFXY REFUSED: promoted-cache cell count moved since stamping'
  Write-Output ("  expected $wantPromCells cells, found $nprom")
  Write-Output '  The produce promoted cache changed; re-verify plan-time provenance before arming.'
  exit 1
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
Write-Output ("PINBU_MINFXY runs -> $runsReal (volume $runsQual free ${diskGB}GB)")
if ($freeGB -lt 30) { Refuse "only $freeGB GB RAM free (need >= 30)" }
if ($diskGB -lt 10) { Refuse "only $diskGB GB free on the runs volume (need >= 10)" }

# -- FRESH run dir -----------------------------------------------------------
# The harness RESUMES from pinbu_wave_results.jsonl, so a stale partial would be
# read back as "already measured" and silently shrink the wave.  Removing it is
# safe here because nothing has been merged yet; after a real partial run, do NOT
# re-run this launcher -- re-invoke the bat, which resumes.
# Only the wave's OWN subdir is removed; runs\ itself (possibly a junction) is not.
Remove-Item (Join-Path $k 'runs\pinbu_wave_minfxy_r1') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_minfxy_r1_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_minfxy_r1_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$bat = Join-Path $k 'run_pinbu_wave_minfxy_r1_199.bat'
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("PINBU_MINFXY Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output 'PINBU_MINFXY LAUNCH FAILED (non-zero ReturnValue)'; exit 1 }

Start-Sleep -Seconds 60
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'pinbu_wave' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'pinbu_wave_minfxy_r1_out.log'
Write-Output ("PINBU_MINFXY armed: hashes_ok=True python=$n master=$mc2 freeRAM=${freeGB}GB" +
              "  log=$log  rc=$k\pinbu_wave_minfxy_r1_rc.txt")
if (Test-Path $log) {
  Write-Output '--- log header ---'
  Get-Content $log -TotalCount 40
} else {
  Write-Output '--- log not created yet ---'
}
