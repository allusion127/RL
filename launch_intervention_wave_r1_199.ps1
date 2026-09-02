# launch_intervention_wave_r1_199.ps1 -- runs ON 199.  Arms CAMPAIGN A ROUND 1
# (Causal Move Atlas): 800 paired single-move interventions over the five F_xy
# frontier cells, 20 parents x 8 balanced moves each, MAS_OUT of every converged
# chain RETAINED so the F_xy parser labels it.
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\USER\lpopt_work\kit_frontier\launch_intervention_wave_r1_199.ps1"
#
# TWO KITS, ONE LAUNCH.  Asset routing is PER LIBRARY (ProduceDriver
# _run_library_id / resolver.build_case_resolver): a paramA cell resolves
# against data\design\package with the registry alias bridge and %GEN_DIM dims
# (40,42), a ga80 cell against FEASIBLE_PACKAGE with LIBRARY_DIMS (83,85).  The
# ablation runner builds ONE resolver from ONE --package, so it shares that
# limitation and `intervention_wave.py run` REFUSES a mixed selection.  The .bat
# therefore makes two sequential invocations, one per kit -- two resolver kits,
# one deployment, one MASTER queue.
#
# LAUNCH MECHANISM: Invoke-CimMethod Win32_Process Create with a LITERAL path to
# cmd.exe (schtasks no-ops silently on this fleet -- the TRIPLE r2 / FXYERA1
# family precedent).
#
# GATES, all REFUSE rather than stack or run on the wrong inputs:
#   BUSY     no master4.0m4_r1, and no lpopt / ablation / batchswap / mesh /
#            produce_fxyera_r1 / intervention python process.  The F_xy-era
#            produce round shares this box and this store; two MASTER queues on
#            one box is how a wave silently loses its cadence.
#   PLAN     sha256 of data\design\intervention_wave_r1.json must equal the value
#            the pre-registration froze (s9).  A wrong or truncated plan is
#            cheaper to catch here than 800 chains later.
#   CODE     intervention_wave.py sha256 -- the artefact the registration hashed.
#            ablation_wave.py must ALSO be present and unmodified: this wave
#            imports its enumerator / annotator / runner / kit builder.
#   STORE    sha256 of records.parquet -- the plan was deduped against THIS
#            store, and a different one mis-dedups (a re-run of an existing
#            label) and mis-ranks the parents.
#   FUEL     sha256 of fuel_types.parquet -- the level-3 pair_ecore restart
#            fallback and the paramA e_core both depend on it, and without it
#            the HGD569 cell cannot resolve at all.
#   ROUTING  data\design\package registry + synth_decks (HGD569 has no packaged
#            template deck; its reload deck is synthesized and cached) AND
#            FEASIBLE_PACKAGE for the three ga80 cells.
#   DISK     >= 25 GB free.  Retention costs ~9.5 MB/chain x 800 ~ 7.6 GB.
#   MODEL    data\models\s1i present (the blind scores were taken against it on
#            the coordinator; the run itself never loads a checkpoint).
#
#   Pre-registration: data/reports/intervention_wave_r1_prereg_20260829.md
#
# Nothing here touches runs\produce_fxyera_r1, runs\ablation_1move_T6T4,
# runs\batchswap_enum_T6T4, any other runs\ dir, or boxes 198 / 181 / 238.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

function Refuse($msg) { Write-Output ("INTERV1 REFUSED: " + $msg); exit 1 }

function Assert-Sha($path, $want, $label) {
  if (-not (Test-Path $path)) { Refuse "$label not found ($path)" }
  $got = (Get-FileHash -Algorithm SHA256 $path).Hash
  if ($got -ne $want) {
    Write-Output "INTERV1 REFUSED: $label sha256 mismatch"
    Write-Output "  expected $want"
    Write-Output "  found    $got"
    exit 1
  }
}

# -- busy gate ---------------------------------------------------------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|mesh|produce_fxyera_r1|intervention' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Refuse "box busy (python=$busy master=$mc) -- produce_fxyera_r1 may still be running"
}

# -- plan / code / store / fuel hash gates (pre-registration s9) --------------
Assert-Sha (Join-Path $k 'data\design\intervention_wave_r1.json') `
  'F82CE02943893D5132FFEC9321ADFA1757C3CF6DC30624CF392E93C2D86FE20D' 'plan'
Assert-Sha (Join-Path $k 'intervention_wave.py') `
  '4D545E814A050953703769A684019F855D9C1944F16EE5D782E5B4B0AA88FDC6' 'intervention_wave.py'
Assert-Sha (Join-Path $k 'ablation_wave.py') `
  '1B94C7128F41685B6B3852527AD8FF6625414F781009AD1ED9F17CAE5F9280C1' 'ablation_wave.py'
Assert-Sha (Join-Path $k 'data\store\records.parquet') `
  '255F0E41707CB4EF64D843FD19DB81531C12AB3A969F6F8F06C87E0AF5561A51' 'store'  # re-stamped 2026-08-30 after min_fxy r1 merge (75,893 rows)
Assert-Sha (Join-Path $k 'data\store\fuel_types.parquet') `
  'FC73AD29741815612C86D91DF746258D20BF9513652A93EA388924B081F78137' 'fuel_types'

if (-not (Test-Path (Join-Path $k 'mine_policy_corpus.py'))) {
  Refuse "mine_policy_corpus.py not at the kit root (the annotator lives there)" }
if (-not (Test-Path (Join-Path $k 'data\models\s1i\ensemble.json'))) {
  Refuse "data\models\s1i not on the kit" }

# -- routing: BOTH packages, because this wave spans both libraries -----------
$pkgA = Join-Path $k 'data\design\package'
$pkgG = Join-Path $k 'FEASIBLE_PACKAGE'
if (-not (Test-Path (Join-Path $pkgA 'registry.json'))) {
  Refuse "data\design\package\registry.json missing (paramA alias routing)" }
if (-not (Test-Path (Join-Path $pkgA 'lib\MAS_XSL'))) {
  Refuse "data\design\package\lib\MAS_XSL missing (library dims are read off it)" }
if (-not (Test-Path (Join-Path $pkgG 'lib\MAS_XSL'))) {
  Refuse "FEASIBLE_PACKAGE\lib\MAS_XSL missing (the three ga80 cells cannot run)" }
# HGD569 has no packaged template deck; its reload deck is SYNTHESIZED and
# cached under synth_decks\<pair>\.  Without this root every HGD569 chain dies
# as a MissingCaseAssetError.
if (-not (Test-Path (Join-Path $k 'data\design\synth_decks'))) {
  Refuse "data\design\synth_decks missing (HGD569 f125 needs a synthesized deck)" }
# The restart the HGD569 parents were themselves labelled on.  The run's own
# restart-provenance gate checks this too, but failing here is free.
if (-not (Test-Path (Join-Path $pkgA 'bases\P0_P1\MAS_RST.APRQ_11_0705.02'))) {
  Refuse "bases\P0_P1\MAS_RST.APRQ_11_0705.02 missing (HGD569 pair_ecore restart)" }
if (-not (Test-Path (Join-Path $pkgA 'bases\T6_T4\MAS_RST.APRQ_10_0615.11'))) {
  Refuse "bases\T6_T4\MAS_RST.APRQ_10_0615.11 missing (T6_T4 native restart)" }

# -- disk gate: retained MAS_OUT is the deliverable, and it costs GB ----------
$drive = (Get-Item $k).PSDrive.Name
$freeGB = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
if ($freeGB -lt 25) {
  Refuse "only $freeGB GB free on ${drive}: (need >= 25 GB; ~7.6 GB of retained MAS_OUT plus headroom)" }

$bat = Join-Path $k 'run_intervention_wave_r1_199.bat'
if (-not (Test-Path $bat)) { Refuse "run_intervention_wave_r1_199.bat not at the kit root" }

# -- FRESH run-dir -----------------------------------------------------------
# A stale partial would be read back as "already done" by the per-cell jsonl
# resume and would silently shrink the wave (ablation launcher precedent).
# NOTE: this is why a RESUME after a crash must NOT be done by re-arming; run
# the .bat directly, which keeps runs\intervention_wave_r1 and re-runs only the
# harness-failed chains.
Remove-Item (Join-Path $k 'runs\intervention_wave_r1') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'intervention_wave_r1_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'intervention_wave_r1_out.log') -Force -EA SilentlyContinue

# -- WMI literal-path launch -------------------------------------------------
$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("INTERV1 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output "INTERV1 LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'intervention_wave' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'intervention_wave_r1_out.log'
Write-Output ("INTERV1 armed: plan_sha_ok=True code_sha_ok=True store_sha_ok=True fuel_sha_ok=True freeGB=$freeGB python=$n master=$mc2  log=$log  rc=$k\intervention_wave_r1_rc.txt")
if (Test-Path $log) { Write-Output '--- log header ---'; Get-Content $log -TotalCount 60 }
else { Write-Output '--- log not created yet ---' }
