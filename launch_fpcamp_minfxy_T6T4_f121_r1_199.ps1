# launch_fpcamp_minfxy_T6T4_f121_r1_199.ps1 -- runs ON 199.  Arms the PROGRAMME'S
# FIRST min_fxy campaign: T6_T4 / feed 121 / paramA, champion s1j (v8), objective
# F_xy (MASTER FXYP) with the hard limit F_xy <= 1.65 and F_r <= 1.55 kept as a
# constraint.  Budget 100 (12 waves x 8 + 4 reserve).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_fpcamp_minfxy_T6T4_f121_r1_199.ps1"
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet -- same as the TRIPLE r2 family).
#
# GATES, all REFUSE rather than stack or run on the wrong inputs:
#   BUSY    no master4.0m4_r1, no lpopt/ablation/batchswap/mesh python.  The
#           F_xy-era produce round (`lpopt produce --input produce_fxyera_r1_199`)
#           matches that same 'lpopt' pattern, so this REFUSES while it drains;
#           an explicit named check follows so the refusal message says WHICH.
#   DECK    sha256 must equal the value the pre-registration froze.
#   MODEL   data\models\s1j present, DONE marker, cond_schema v8.
#   STORE   sha256 of records.parquet must equal the 2026-08-30 canonical store
#           -- the 746 f_xy labels at this cell ARE the elite pool
#           (`_store_elites` ranks by measured F_xy under min_fxy); a stale store
#           starves it and the round silently ranks on nothing.
#   ROUTING data\design\package registry + the T6_T4 level-3 restart.
#   HARVEST the deck must still say `harvest_maps = true` -- min_fxy has no
#           labels without it, and the campaign driver hard-refuses.  Belt and
#           braces on top of the deck hash.
#
#   Pre-registration: data/reports/minfxy_T6T4_f121_r1_prereg_20260829.md
#
# LAUNCH-TIME MODEL SUBSTITUTION: APPLIED 2026-08-30.  The f_xy head was promoted to
# data\models\s1j (results data/reports/fxy_head_results_arm3_20260829.md: G1/G2'/G3'
# PASS, G4 FAIL -> the head SIGMA is barred from serving, the head MEAN serves).  The
# deck's [model] model_dir line was edited (that line ONLY), the deck re-hashed, and
# $want / $modelName below updated.  Nothing else moved.  The deck's OTHER comment
# lines still describe s1i -- untouched on purpose: prereg SS5.3 lets exactly one deck
# line change.  The stamped record is minfxy_T6T4_f121_r1_prereg_20260829.md SS9.1.
#
# Nothing here touches runs/fpcamp_minfr_triple_f125_r2, runs/produce_fxyera_r1,
# any other runs/ dir, or boxes 198 / 181 / 238.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'
$modelName = 's1j'          # LAUNCH-TIME substitution APPLIED 2026-08-30 (was 's1i')

# -- busy gate ---------------------------------------------------------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("MINFXY1 REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

# -- named busy gate: the F_xy-era produce round must have DRAINED ------------
$prod = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'produce_fxyera_r1' }).Count
if ($prod -gt 0) {
  Write-Output ("MINFXY1 REFUSED: produce_fxyera_r1 is still running (python=$prod) -- it owns the box and it is the source of this cell's f_xy labels"); exit 1
}
$prc = Join-Path $k 'produce_fxyera_r1_rc.txt'
$plog = Join-Path $k 'produce_fxyera_r1_out.log'
if ((Test-Path $plog) -and -not (Test-Path $prc)) {
  Write-Output "MINFXY1 REFUSED: produce_fxyera_r1 started but wrote no rc file -- it is either running or died mid-round; resolve it before stacking a campaign"; exit 1
}

# -- deck hash gate ----------------------------------------------------------
$deck = Join-Path $k 'fpcamp_minfxy_T6T4_f121_r1_199.inp'
if (-not (Test-Path $deck)) { Write-Output "MINFXY1 REFUSED: deck not found ($deck)"; exit 1 }
$want = 'BEF3519E720FE1F94FE1448EF3046FCE1EB15BD94DBA5DD1F4E9B2F3976C95C9'   # s1j deck (was 4AF8B021... for s1i), 16,957 B
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {
  Write-Output "MINFXY1 REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  Write-Output "  (if this is the intended s1j substitution, update `$want here and in the prereg SS9)"
  exit 1
}

# -- harvest_maps gate: min_fxy cannot label itself without it ----------------
if (-not (Select-String -Path $deck -Pattern '^\s*harvest_maps\s*=\s*true' -Quiet)) {
  Write-Output "MINFXY1 REFUSED: deck does not set [verify] harvest_maps = true -- F_xy is parsed from the final cycle MAS_OUT, which only survives because harvest_maps forces keep_success"; exit 1
}

# -- model gate: present, DONE, AND v8 ---------------------------------------
$md  = Join-Path $k ('data\models\' + $modelName)
$ens = Join-Path $md 'ensemble.json'
if (-not (Test-Path $ens)) { Write-Output "MINFXY1 REFUSED: data\models\$modelName not on the kit"; exit 1 }
if (-not (Test-Path (Join-Path $md 'DONE'))) {
  Write-Output "MINFXY1 REFUSED: data\models\$modelName carries no DONE marker"; exit 1 }
$meta = Get-ChildItem -Path $md -Filter 'meta.json' -Recurse -EA SilentlyContinue | Select-Object -First 1
if (-not $meta) { Write-Output "MINFXY1 REFUSED: $modelName member meta.json missing"; exit 1 }
$schema = (Get-Content $meta.FullName -Raw | ConvertFrom-Json).cond_schema
if ($schema -ne 'v8') {
  Write-Output "MINFXY1 REFUSED: $modelName cond_schema is '$schema', expected 'v8' (the deck declares v8; a mismatch is a silent serving error)"; exit 1 }

# -- store gate: the canonical, f_xy-labelled store ---------------------------
$st = Join-Path $k 'data\store\records.parquet'
if (-not (Test-Path $st)) { Write-Output "MINFXY1 REFUSED: store missing"; exit 1 }
# STORE PIN -- RE-STAMP REQUIRED AFTER THE pinbu PHASE-2 MERGE.  This is the store as
# of 2026-08-30 (75,793 rows, post e_core-backfill); the pre-backfill pin
# 0334E2D2E303CD8E82373861603F82A57054FC2FC6139F84C360822580644D9D would refuse today.
# The phase-2 `pinbu_wave` merge (minfxy prereg SS5.4) WILL append rows and change this
# hash: when it lands, re-hash data/store/records.parquet and update BOTH this line and
# the store row of minfxy_T6T4_f121_r1_prereg_20260829.md SS9.1, or the launcher refuses.
$wantStore = '72516916F5D59A738BA95CE2A7D56F0F2E9F514E61DD654BE4BE6127D175CE5D'  # re-stamped 2026-08-30 after pinbu phase-2 patch (32 max_pin_burnup cells)
$gotStore  = (Get-FileHash -Algorithm SHA256 $st).Hash
if ($gotStore -ne $wantStore) {
  Write-Output "MINFXY1 REFUSED: store sha256 mismatch (the 746 f_xy labels at T6_T4/f121 ARE the elite pool under min_fxy)"
  Write-Output "  expected $wantStore"
  Write-Output "  found    $gotStore"
  exit 1
}

$bat = Join-Path $k 'run_fpcamp_minfxy_T6T4_f121_r1_199.bat'
if (-not (Test-Path $bat)) { Write-Output "MINFXY1 REFUSED: run bat not at the kit root"; exit 1 }

# -- paramA routing: the design package, NOT FEASIBLE_PACKAGE ------------------
$pkg = Join-Path $k 'data\design\package'
if (-not (Test-Path (Join-Path $pkg 'registry.json'))) {
  Write-Output "MINFXY1 REFUSED: data\design\package\registry.json missing (paramA routing)"; exit 1 }
$base = Join-Path $pkg 'bases\T6_T4\MAS_RST.APRQ_10_0615.11'
if (-not (Test-Path $base)) {
  Write-Output "MINFXY1 REFUSED: bases\T6_T4\MAS_RST.APRQ_10_0615.11 missing (expected T6_T4 restart source)"; exit 1 }

# -- disk: harvest_maps retains the final cycle dir of every converged chain ---
$free = (Get-PSDrive C).Free / 1GB
if ($free -lt 20) { Write-Output ("MINFXY1 REFUSED: only {0:N1} GB free on C: (need >= 20 for 100 retained chains)" -f $free); exit 1 }

# -- FRESH run-dir -----------------------------------------------------------
Remove-Item (Join-Path $k 'runs\fpcamp_minfxy_t6t4_f121_r1') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfxy_t6t4_f121_r1_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'fpcamp_minfxy_t6t4_f121_r1_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("MINFXY1 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "MINFXY1 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'fpcamp_minfxy_T6T4_f121_r1' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'fpcamp_minfxy_t6t4_f121_r1_out.log'
Write-Output ("MINFXY1 armed: deck_sha_ok=True store_sha_ok=True model=$modelName schema=v8 python=$n master=$mc2  log=$log")
# The F_xy SOURCE line is the one to read: 'proxy' = s1i arm, 'head' = s1j arm.
if (Test-Path $log) {
  Write-Output '--- F_xy source banner ---'
  Select-String -Path $log -Pattern 'min_fxy objective|F_xy PROXY|predict_fxy' | ForEach-Object { $_.Line }
  Write-Output '--- log header ---'
  Get-Content $log -TotalCount 50
} else { Write-Output '--- log not created yet ---' }
