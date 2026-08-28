# launch_pinbu_wave_199.ps1 -- runs ON 199.  Arms the MEASURED PIN-BURNUP WAVE
# (44 stored cores replayed with enable_pin_burnup=True).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_pinbu_wave_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe.  schtasks no-ops silently on this fleet
# (data/reports/ab2_addendum_ADF_20260810.md section 3), so it is not used.
#
# BUSY GATE: REFUSES rather than stacks (v520_b2 precedent).  199 was idle at
# pre-registration; if anything is running, that assumption is wrong.
#
# HASH GATES: the deck AND the plan must be the artifacts the pre-registration
# hashed.  The plan carries every pinned prediction this wave will be scored
# against -- a swapped plan would silently un-register the experiment.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate ---------------------------------------------------------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|pinbu' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("PINBU REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- hash gates --------------------------------------------------------------
$gates = @(
  @{ Path = Join-Path $k 'pinbu_wave_199.inp';
     Want = '03EEADE4AF84DF8C74BF00513869D84BB34A3EA21DDC43F10E6783A0D1541DDA';
     Name = 'deck' },
  @{ Path = Join-Path $k 'data\reports\pinbu_wave_prereg_20260820.json';
     Want = '45A0A24D12BA251E3917A01F7E1B69AF9C18119DFA613A205ED71B2BA9D8EA0E';
     Name = 'prereg plan' },
  @{ Path = Join-Path $k 'pinbu_wave.py';
     Want = 'CAF07B25E291BE6A6B52657981D05560D2DBD78A403B94B31F3A8D4D166FFD1D';
     Name = 'harness' },
  @{ Path = Join-Path $k 'run_pinbu_wave_199.bat';
     Want = '294290B21940E1009C3113D0505805E6D185DE1F90078B2BD708EE34EB192B53';
     Name = 'run bat' }
)
foreach ($g in $gates) {
  if (-not (Test-Path $g.Path)) {
    Write-Output ("PINBU REFUSED: " + $g.Name + " not found (" + $g.Path + ")"); exit 1
  }
  $got = (Get-FileHash -Algorithm SHA256 $g.Path).Hash
  if ($got -ne $g.Want) {
    Write-Output ("PINBU REFUSED: " + $g.Name + " sha256 mismatch")
    Write-Output ("  expected " + $g.Want)
    Write-Output ("  found    " + $got)
    exit 1
  }
}

# -- precondition gates ------------------------------------------------------
# The plan pinned its predictions against THIS store snapshot; the run itself
# reads the store only for fuel_types.parquet, but a different records.parquet
# means the kit has drifted from the coordinator and the merge target is unclear.
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output 'PINBU REFUSED: store missing'; exit 1 }
$stLen = (Get-Item $st).Length
if ($stLen -ne 22196891) {
  Write-Output "PINBU REFUSED: store is not the pinned snapshot (bytes=$stLen, want 22196891)"; exit 1
}
$ft = Join-Path $k 'data\store\fuel_types.parquet'
if (-not (Test-Path $ft)) { Write-Output 'PINBU REFUSED: fuel_types.parquet missing'; exit 1 }
# ga80 (N1_N2/f113, E1_E2/f109) and paramA (both hgd569/f125 cells) packages
foreach ($p in @('FEASIBLE_PACKAGE\bases\N1_N2', 'FEASIBLE_PACKAGE\bases\E1_E2',
                 'data\design\package\cores', 'data\design\package\lib')) {
  if (-not (Test-Path (Join-Path $k $p))) {
    Write-Output "PINBU REFUSED: $p missing"; exit 1
  }
}
if (-not (Test-Path 'C:/DeCART_MASTER/BIN/master4.0m4_r1.exe')) {
  Write-Output 'PINBU REFUSED: MASTER executable not found'; exit 1
}
# 12 concurrent chains: refuse on a box that cannot hold them.
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB,1)
$diskGB = [math]::Round((Get-PSDrive C).Free/1GB,1)
if ($freeGB -lt 30) { Write-Output "PINBU REFUSED: only $freeGB GB RAM free (need >= 30)"; exit 1 }
if ($diskGB -lt 15) { Write-Output "PINBU REFUSED: only $diskGB GB disk free (need >= 15)"; exit 1 }

# -- FRESH run dir -----------------------------------------------------------
# The harness RESUMES from pinbu_wave_results.jsonl, so a stale partial would be
# read back as "already measured" and silently shrink the wave.  Removing it is
# safe here because nothing has been merged yet; after a real partial run, do
# NOT re-run this launcher -- re-invoke the bat, which resumes.
Remove-Item (Join-Path $k 'runs\pinbu_wave') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$bat = Join-Path $k 'run_pinbu_wave_199.bat'
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("PINBU Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output 'PINBU LAUNCH FAILED (non-zero ReturnValue)'; exit 1 }

Start-Sleep -Seconds 60
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'pinbu_wave' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'pinbu_wave_out.log'
Write-Output ("PINBU armed: hashes_ok=True python=$n master=$mc2 freeRAM=${freeGB}GB" +
              "  log=$log  rc=$k\pinbu_wave_rc.txt")
if (Test-Path $log) {
  Write-Output '--- log header ---'
  Get-Content $log -TotalCount 40
} else {
  Write-Output '--- log not created yet ---'
}
