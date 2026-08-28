# launch_fpcamp_mt1_e50_f113_199.ps1 -- runs ON 199.  P3b multi-type anchor campaign 1/1:
# E3_E1_E2 / feed 113 / ga80, champion s1i (v8).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_mt1_e50_f113_199.ps1"
#
# BUSY GATE: REFUSES rather than stacks.
# DECK GATE: the deck must be the one the pre-registration hashed.
# MODEL GATE: s1i present AND cond_schema v8 -- a v7/v6b checkpoint would either
#             abort or score the graded core as if a rung were not there.
# STORE GATE: the canonical store carrying the DONOR rows for elite_seed_cases.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("MT1_e50_f113 REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

$deck = Join-Path $k 'fpcamp_mt1_e50_f113_199.inp'
if (-not (Test-Path $deck)) { Write-Output "MT1_e50_f113 REFUSED: deck not found ($deck)"; exit 1 }
$want = 'F4CFE4DDB73868F105284B1EA986B0B7F7B6192198A2BC830DC5AD07D7E277A9'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "MT1_e50_f113 REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

$ens = Join-Path $k 'data\models\s1i\ensemble.json'
if (-not (Test-Path $ens)) { Write-Output "MT1_e50_f113 REFUSED: data\models\s1i not on the kit"; exit 1 }
if (-not (Test-Path (Join-Path $k 'data\models\s1i\DONE'))) {
  Write-Output "MT1_e50_f113 REFUSED: data\models\s1i carries no DONE marker"; exit 1 }
$meta = Join-Path $k 'data\models\s1i\member_20260716\meta.json'
if (-not (Test-Path $meta)) { Write-Output "MT1_e50_f113 REFUSED: s1i member meta.json missing"; exit 1 }
$schema = (Get-Content $meta -Raw | ConvertFrom-Json).cond_schema
if ($schema -ne 'v8') {
  Write-Output "MT1_e50_f113 REFUSED: s1i cond_schema is '$schema', expected 'v8'"; exit 1 }

$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "MT1_e50_f113 REFUSED: store missing"; exit 1 }
$wantStore = 'ECEF9FAE27C9E58DAC9B8ED2095051604BFBBFB9ED52AF2D7AB26C30B8292C9B'
$gotStore  = (Get-FileHash -Algorithm SHA256 $st).Hash
if ($gotStore -ne $wantStore) {
  Write-Output "MT1_e50_f113 REFUSED: store sha256 mismatch (the DONOR rows for elite_seed_cases must be present)"
  Write-Output "  expected $wantStore"
  Write-Output "  found    $gotStore"
  exit 1
}

$bat = Join-Path $k 'run_fpcamp_mt1_e50_f113_199.bat'
if (-not (Test-Path $bat)) { Write-Output "MT1_e50_f113 REFUSED: run bat not at the kit root"; exit 1 }

$pkg = Join-Path $k 'FEASIBLE_PACKAGE'
if (-not (Test-Path (Join-Path $pkg 'manifest.csv'))) {
  Write-Output "MT1_e50_f113 REFUSED: FEASIBLE_PACKAGE\manifest.csv missing (ga80 routing)"; exit 1 }

Remove-Item (Join-Path $k 'runs\fpcamp_mt1_e50_f113') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_mt1_e50_f113_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_mt1_e50_f113_out.log') -Force -EA SilentlyContinue

$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("MT1_e50_f113 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "MT1_e50_f113 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_mt1_e50_f113_199' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_mt1_e50_f113_out.log'
Write-Output ("MT1_e50_f113 armed: deck_sha_ok=True store_sha_ok=True schema=v8 python=$n master=$mc2  log=$log")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 50 }
else { Write-Output '--- log not created yet ---' }
