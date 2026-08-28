# launch_fpcamp_HGD569_f125_SEEDCTL_199.ps1 -- runs ON 199.  Arms the R-SEED
# control: 2-type P6253Z1G06N24_P6253Z2G10N24/f125, champion s1h (v7), elite
# pool seeded by the pair's OWN store-resident converged output (no reship --
# already verified present).  De-confounds the 3-type -0.0364 headline
# (tripletype_f125_results_20260817.md §8).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_HGD569_f125_SEEDCTL_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet).
#
# BUSY GATE: REFUSES rather than stacks.
# DECK GATE: the deck must be the one the pre-registration hashed.
# MODEL GATE: s1h must be present AND must be a v7 (18-global) checkpoint.
# STORE GATE: the store already on 199 must still carry >= 129 converged rows
#             for the donor pair (73 @ f125, 56 @ f109) -- the SAME population
#             elite_seed_cases fed the 3-type run.  Pinned by SHA256 rather than
#             re-probed live so any drift (another run touching the store)
#             REFUSES instead of silently seeding from a different population.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate (includes meshcalc / scoping-mesh / prior-campaign leftovers) --
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("SEEDCTL REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'fpcamp_minfr_HGD569_f125_SEEDCTL_199.inp'
if (-not (Test-Path $deck)) { Write-Output "SEEDCTL REFUSED: deck not found ($deck)"; exit 1 }
$want = '8E5D3A75F4200797F4C6A59D4022A8F665C980921A54F1C44D8A8C8AAC24B341'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "SEEDCTL REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

# -- model gate: present AND v7 ----------------------------------------------
$ens = Join-Path $k 'data\models\s1h\ensemble.json'
if (-not (Test-Path $ens)) { Write-Output "SEEDCTL REFUSED: data\models\s1h not on the kit"; exit 1 }
if (-not (Test-Path (Join-Path $k 'data\models\s1h\DONE'))) {
  Write-Output "SEEDCTL REFUSED: data\models\s1h carries no DONE marker"; exit 1 }
$meta = Join-Path $k 'data\models\s1h\member_20260716\meta.json'
if (-not (Test-Path $meta)) { Write-Output "SEEDCTL REFUSED: s1h member meta.json missing"; exit 1 }
$schema = (Get-Content $meta -Raw | ConvertFrom-Json).cond_schema
if ($schema -ne 'v7') {
  Write-Output "SEEDCTL REFUSED: s1h cond_schema is '$schema', expected 'v7'"; exit 1 }

# -- store gate: the donor-carrying store already resident on 199 -------------
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "SEEDCTL REFUSED: store missing"; exit 1 }
$wantStore = '4BE89C61856CCD2B41DDDC9B546B1217AC95BDBF019225CDAFFFFE73C5DA8ADD'
$gotStore  = (Get-FileHash -Algorithm SHA256 $st).Hash
if ($gotStore -ne $wantStore) {
  Write-Output "SEEDCTL REFUSED: store sha256 mismatch (the donor rows for the elite pool must be present unmodified)"
  Write-Output "  expected $wantStore"
  Write-Output "  found    $gotStore"
  exit 1
}

$bat = Join-Path $k 'run_fpcamp_minfr_HGD569_f125_SEEDCTL_199.bat'
if (-not (Test-Path $bat)) { Write-Output "SEEDCTL REFUSED: run bat not at the kit root"; exit 1 }

# -- paramA routing: the design package, NOT FEASIBLE_PACKAGE ------------------
$pkg = Join-Path $k 'data\design\package'
if (-not (Test-Path (Join-Path $pkg 'registry.json'))) {
  Write-Output "SEEDCTL REFUSED: data\design\package\registry.json missing (paramA routing)"; exit 1 }
$base = Join-Path $pkg 'bases\P0_P1\MAS_RST.APRQ_11_0705.02'
if (-not (Test-Path $base)) {
  Write-Output "SEEDCTL REFUSED: bases\P0_P1\MAS_RST.APRQ_11_0705.02 missing (expected level-3 restart source)"; exit 1 }

# -- FRESH run-dir -----------------------------------------------------------
Remove-Item (Join-Path $k 'runs\fpcamp_minfr_hgd569_f125_seedctl') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_hgd569_f125_seedctl_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_hgd569_f125_seedctl_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("SEEDCTL Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "SEEDCTL LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_minfr_HGD569_f125_SEEDCTL' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_minfr_hgd569_f125_seedctl_out.log'
Write-Output ("SEEDCTL armed: deck_sha_ok=True store_sha_ok=True schema=v7 python=$n master=$mc2  log=$log")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 50 }
else { Write-Output '--- log not created yet ---' }
