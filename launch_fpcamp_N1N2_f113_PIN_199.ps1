# launch_fpcamp_N1N2_f113_PIN_199.ps1 -- runs ON 199. Arms the PIN-GATED
# RE-CAMPAIGN: N1_N2/f113, champion s1i (v8), minfr_pin_bu_limit=78.0 active
# (5th gate). The 2026-08-16 campaign's own winner (F_r 1.4961) measures pin
# 86.189 -- FAIL. This run adds the pin gate at the same cell.
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_N1N2_f113_PIN_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet).
#
# BUSY GATE: REFUSES rather than stacks.
# DECK GATE: the deck must be the one the pre-registration hashed.
# MODEL GATE: s1i must be present AND must be a v8 (member meta.json) checkpoint.
# STORE GATE: the store already on 199 must match the local canonical store's
#             sha256 exactly (post-pinbu-merge, 74,597 rows) -- pinned rather
#             than re-probed live so any drift REFUSES instead of silently
#             running on a different population.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# -- busy gate (includes meshcalc / scoping-mesh / prior-campaign leftovers) --
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("PIN REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'fpcamp_minfr_N1N2_f113_PIN_199.inp'
if (-not (Test-Path $deck)) { Write-Output "PIN REFUSED: deck not found ($deck)"; exit 1 }
$want = 'D637EB0834F1DF2C7ACCCF8BB43B8484E219EEC633CC10F541BD0BFF9023089C'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "PIN REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}

# -- model gate: present AND v8 -----------------------------------------------
$ens = Join-Path $k 'data\models\s1i\ensemble.json'
if (-not (Test-Path $ens)) { Write-Output "PIN REFUSED: data\models\s1i not on the kit"; exit 1 }
if (-not (Test-Path (Join-Path $k 'data\models\s1i\DONE'))) {
  Write-Output "PIN REFUSED: data\models\s1i carries no DONE marker"; exit 1 }
$meta = Join-Path $k 'data\models\s1i\member_20260716\meta.json'
if (-not (Test-Path $meta)) { Write-Output "PIN REFUSED: s1i member meta.json missing"; exit 1 }
$schema = (Get-Content $meta -Raw | ConvertFrom-Json).cond_schema
if ($schema -ne 'v8') {
  Write-Output "PIN REFUSED: s1i cond_schema is '$schema', expected 'v8'"; exit 1 }

# -- store gate: byte-identical to the local canonical store ------------------
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "PIN REFUSED: store missing"; exit 1 }
$wantStore = '76346DFD6324BDA8D48117CA14FD344CA5BE29AF76C0E01BB31A96267FA336FF'
$gotStore  = (Get-FileHash -Algorithm SHA256 $st).Hash
if ($gotStore -ne $wantStore) {
  Write-Output "PIN REFUSED: store sha256 mismatch (must be the post-pinbu-merge 74,597-row canonical store)"
  Write-Output "  expected $wantStore"
  Write-Output "  found    $gotStore"
  exit 1
}

$bat = Join-Path $k 'run_fpcamp_minfr_N1N2_f113_PIN_199.bat'
if (-not (Test-Path $bat)) { Write-Output "PIN REFUSED: run bat not at the kit root"; exit 1 }

# -- FRESH run-dir -------------------------------------------------------------
Remove-Item (Join-Path $k 'runs\fpcamp_minfr_N1N2_f113_pin') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_N1N2_f113_pin_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfr_N1N2_f113_pin_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch ---------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("PIN Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "PIN LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_minfr_N1N2_f113_PIN' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_minfr_N1N2_f113_pin_out.log'
Write-Output ("PIN armed: deck_sha_ok=True store_sha_ok=True schema=v8 python=$n master=$mc2  log=$log")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 50 }
else { Write-Output '--- log not created yet ---' }
