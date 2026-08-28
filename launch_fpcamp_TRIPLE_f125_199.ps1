# launch_fpcamp_TRIPLE_f125_199.ps1 -- runs ON 199.  Arms the FIRST
# 3-fresh-type (graded) campaign: S3 _ <mid> _ S5 / feed 125 / paramA, champion
# s1h (v7).  Successor to the 2-type f125 campaign (closed, joint-clean F_r
# 1.6357, 4 of 5 gates clear).  Pure F_r minimisation under five gates, no
# cycle band.
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_TRIPLE_f125_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet).
#
# BUSY GATE: REFUSES rather than stacks.
# DECK GATE: the deck must be the one the pre-registration hashed.
# MODEL GATE: s1h must be present AND must be a v7 (18-global) checkpoint -- an
#             s1g/v6b model cannot encode a 3-type core and would either abort or,
#             worse, score the third type as if it were not there.
# STORE GATE: the refreshed canonical store must be the one that carries the
#             DONOR rows, or the elite pool is empty and the campaign is a
#             cold random search wearing a graded costume.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate (includes meshcalc / scoping-mesh / prior-campaign leftovers) --
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("TRIPLE125 REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'fpcamp_minfr_TRIPLE_f125_199.inp'
if (-not (Test-Path $deck)) { Write-Output "TRIPLE125 REFUSED: deck not found ($deck)"; exit 1 }
$want = 'B042D49AFC274EA2DA630627D49DD0518205E71EA9E44BC41F90BF39D7E342F1'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "TRIPLE125 REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

# -- model gate: present AND v7 ----------------------------------------------
$ens = Join-Path $k 'data\models\s1h\ensemble.json'
if (-not (Test-Path $ens)) { Write-Output "TRIPLE125 REFUSED: data\models\s1h not on the kit"; exit 1 }
if (-not (Test-Path (Join-Path $k 'data\models\s1h\DONE'))) {
  Write-Output "TRIPLE125 REFUSED: data\models\s1h carries no DONE marker"; exit 1 }
$meta = Join-Path $k 'data\models\s1h\member_20260716\meta.json'
if (-not (Test-Path $meta)) { Write-Output "TRIPLE125 REFUSED: s1h member meta.json missing"; exit 1 }
$schema = (Get-Content $meta -Raw | ConvertFrom-Json).cond_schema
if ($schema -ne 'v7') {
  Write-Output "TRIPLE125 REFUSED: s1h cond_schema is '$schema', expected 'v7' (a non-v7 checkpoint cannot encode a 3-type core)"; exit 1 }

# -- store gate: the canonical, donor-carrying store --------------------------
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "TRIPLE125 REFUSED: store missing"; exit 1 }
$wantStore = 'B5B29460A0C1C1E6AD3E3B2B9F410ED021E306099A91ECEEC8B05775B739CAE7'
$gotStore  = (Get-FileHash -Algorithm SHA256 $st).Hash
if ($gotStore -ne $wantStore) {
  Write-Output "TRIPLE125 REFUSED: store sha256 mismatch (the DONOR rows for elite_seed_cases must be present)"
  Write-Output "  expected $wantStore"
  Write-Output "  found    $gotStore"
  exit 1
}

$bat = Join-Path $k 'run_fpcamp_minfr_TRIPLE_f125_199.bat'
if (-not (Test-Path $bat)) { Write-Output "TRIPLE125 REFUSED: run bat not at the kit root"; exit 1 }

# -- paramA routing: the design package, NOT FEASIBLE_PACKAGE ------------------
$pkg = Join-Path $k 'data\design\package'
if (-not (Test-Path (Join-Path $pkg 'registry.json'))) {
  Write-Output "TRIPLE125 REFUSED: data\design\package\registry.json missing (paramA routing)"; exit 1 }
# level-3 pair_ecore restart source -- the same one both 2-type siblings resolved
# to; the triple's |de| to the pair is 0.009-0.019, so it is the expected pick.
$base = Join-Path $pkg 'bases\P0_P1\MAS_RST.APRQ_11_0705.02'
if (-not (Test-Path $base)) {
  Write-Output "TRIPLE125 REFUSED: bases\P0_P1\MAS_RST.APRQ_11_0705.02 missing (expected level-3 restart source)"; exit 1 }

# -- FRESH run-dir -----------------------------------------------------------
Remove-Item (Join-Path $k 'runs\fpcamp_minfr_triple_f125') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_triple_f125_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_triple_f125_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("TRIPLE125 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "TRIPLE125 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_minfr_TRIPLE_f125' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_minfr_triple_f125_out.log'
Write-Output ("TRIPLE125 armed: deck_sha_ok=True store_sha_ok=True schema=v7 python=$n master=$mc2  log=$log")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 50 }
else { Write-Output '--- log not created yet ---' }
