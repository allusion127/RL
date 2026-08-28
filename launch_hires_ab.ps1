<#
    launch_hires_ab.ps1 — hires A/B arm launcher (design doc
    data/reports/hires_model_ab_design_20260725.md).

    DOES NOT LAUNCH BY DEFAULT.  Without -Go it prints the pre-flight GPU report
    and the exact command for every arm, and exits.  Add -Go to actually launch.

    GPU rule (user directive): GPU 0 is the standing choice, GPU 1 is allowed
    when idle.  NEVER take a GPU another user occupies.  -Go refuses to launch
    onto a GPU whose used memory exceeds -MaxUsedMiB OR whose utilization
    exceeds -MaxUtilPct, unless -Force is given.  Utilization is the binding
    signal on a 97 GB card: a neighbour using 4 GB is harmless, a neighbour
    holding the SMs at 98% is not.

    The [remote] table in lpopt.inp pins gpu = 0.  To run on GPU 1 this script
    writes a TEMPORARY deck copy with the chosen index and passes it via
    --input, so lpopt.inp itself is never modified.

    Usage
      .\launch_hires_ab.ps1                      # dry run: pre-flight + commands
      .\launch_hires_ab.ps1 -Arm B1 -Go          # launch one arm
      .\launch_hires_ab.ps1 -Arm B1,A1,A2 -Go    # launch several, sequentially
      .\launch_hires_ab.ps1 -Gpu 1 -Arm A5 -Go   # pin GPU 1 (checked for idleness)
      .\launch_hires_ab.ps1 -Status 20260725_HHMMSS   # wraps: --ts <ts> status
#>
[CmdletBinding()]
param(
    # Default order puts A2 (highest expected effect) right after the control so
    # the decisive comparison lands early; A5 (the null arm) and A6 (the longest)
    # run last.
    [string[]]$Arm = @('B1','A2','A1','A3','A4','A6','A5'),
    # Default 0: the standing user directive and the [remote] deck both pin GPU 0.
    # 'auto' would drift to whichever card is momentarily idler, which is exactly
    # what the directive forbids -- pass it (or '1') only deliberately.
    [ValidateSet('auto','0','1')][string]$Gpu = '0',
    # The card has 97 GB, and the heaviest arm (A5) needs ~60 GB, so a few GB of
    # someone else's memory is not itself a blocker -- but a busy SM is: two jobs
    # at 98% util halve each other's throughput.  Both are checked.
    [int]$MaxUsedMiB = 20000,
    [int]$MaxUtilPct = 25,
    [switch]$Go,
    [switch]$Force,
    [string]$Status,
    [switch]$Push
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# --- the recipe every arm shares (= champion 20260724_213535 MINUS the freeze) --
# --init-from / --freeze-trunk-cyclen are deliberately absent: the champion
# trained only 6,298 of its 3,159,291 parameters, so its trunk cannot respond to
# any architectural change.  B1 is the trunk-trainable control (design doc §2).
$Common = @(
    '--ensemble','5','--split','S1','--epochs','150','--num-workers','8',
    '--device','auto','--parallel-members','5',
    '--cyclen-physics-prior','--quantile-heads','--quantile-weight','0.2',
    '--promote-max-asm-bu',
    '--distill-targets','data/models/_v5_distill_soft.npz',
    '--distill-weight','0.4','--distill-min-match-frac','0.5',
    '--f-r-rank-weight','0.1','--cyclen-rank-weight','0.25',
    '--map-peak-weight','2.0'
)

$Arms = [ordered]@{
    # control: full-trunk training at the champion's shape
    'B1' = @('--cond-schema','v5','--width','160','--n-blocks','6')
    # A1 multiscale map decoder (high-frequency skip path)
    'A1' = @('--cond-schema','v5','--width','160','--n-blocks','6',
             '--map-decoder','multiscale')
    # A2 diffusion power-map prior as input channels + residual map head
    'A2' = @('--cond-schema','v6_prior','--width','160','--n-blocks','6',
             '--map-prior-residual')
    # A3 band-weighted spectral map loss
    'A3' = @('--cond-schema','v5','--width','160','--n-blocks','6',
             '--map-spectral-weight','0.3')
    # A4 local neighbour-contrast input channels
    'A4' = @('--cond-schema','v6_contrast','--width','160','--n-blocks','6')
    # A5 PURE CAPACITY (the pre-registered null arm): 3.16M -> 13.14M
    'A5' = @('--cond-schema','v5','--width','256','--n-blocks','10',
             '--head-hidden','512')
    # A6 combined structure + mid capacity: 10.34M
    'A6' = @('--cond-schema','v6','--width','224','--n-blocks','8',
             '--head-hidden','384','--map-decoder','multiscale',
             '--map-prior-residual','--map-spectral-weight','0.3')
}

$Notes = [ordered]@{
    'B1' = '3.16M params  ~40 GB  ~40 min   control'
    'A1' = '4.31M params  ~48 GB  ~50 min   arm A1'
    'A2' = '3.17M params  ~41 GB  ~42 min   arm A2  (highest expected effect)'
    'A3' = '3.16M params  ~40 GB  ~43 min   arm A3'
    'A4' = '3.16M params  ~40 GB  ~40 min   arm A4'
    'A5' = '13.14M params ~60 GB  ~2.7 h    arm A5  (NULL hypothesis arm)'
    'A6' = '10.34M params ~55 GB  ~2.1 h    arm A6  (combination)'
}

# The login shell on gpu2-6000 prints an unrelated oneAPI warning on stderr.
# Under Windows PowerShell 5.1 + $ErrorActionPreference='Stop' that alone aborts
# the script, so native ssh calls are made with the preference relaxed and the
# stderr chatter filtered out by shape.
function Invoke-Ssh([string]$RemoteCommand) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $out = & ssh -p 8022 USER@HOST_238 $RemoteCommand 2>$null }
    finally { $ErrorActionPreference = $old }
    return @($out | Where-Object { $_ -and $_.ToString().Trim() })
}

function Invoke-Remote([string[]]$RemoteArgs) {
    & python -m lpopt.remote @RemoteArgs
    if ($LASTEXITCODE -ne 0) { throw "lpopt.remote failed (exit $LASTEXITCODE)" }
}

if ($Status) { Invoke-Remote @('--ts',$Status,'status'); return }

# --------------------------------------------------------------------------- #
# pre-flight: who is on which GPU
# --------------------------------------------------------------------------- #
Write-Host '=== GPU pre-flight (gpu2-6000) ===' -ForegroundColor Cyan
$smi = Invoke-Ssh "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
$used = @{}
$util = @{}
foreach ($line in $smi) {
    $f = $line -split '\s*,\s*'
    if ($f.Count -ge 5 -and $f[0] -match '^\d+$') {
        $used[$f[0]] = [int]$f[2]
        $util[$f[0]] = [int]$f[4]
        Write-Host ("  GPU {0} {1,-28} {2,6}/{3} MiB  util {4}%" -f $f[0],$f[1],$f[2],$f[3],$f[4])
    }
}
if ($used.Count -eq 0) { throw 'cannot read nvidia-smi on gpu2-6000; aborting' }
Write-Host '--- compute processes ---' -ForegroundColor Cyan
$procs = Invoke-Ssh "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader"
if ($procs.Count -eq 0) { Write-Host '  (none)' } else { $procs | ForEach-Object { Write-Host "  $_" } }

if ($Gpu -eq 'auto') {
    $pick = ($used.GetEnumerator() | Sort-Object Value | Select-Object -First 1).Key
    if (-not $pick) { $pick = '0' }
} else { $pick = $Gpu }
Write-Host ""
Write-Host "selected GPU: $pick (used $($used[$pick]) MiB, util $($util[$pick])%)" -ForegroundColor Yellow

$busy = @()
if ($used[$pick] -gt $MaxUsedMiB) { $busy += "memory $($used[$pick]) MiB > $MaxUsedMiB" }
if ($util[$pick] -gt $MaxUtilPct) { $busy += "utilization $($util[$pick])% > $MaxUtilPct%" }
if ($busy.Count -and -not $Force) {
    Write-Host "GPU $pick looks OCCUPIED: $($busy -join '; ')." -ForegroundColor Red
    Write-Host "Another user's job is probably running -- see the process list above." -ForegroundColor Red
    Write-Host "Wait for it to finish, pick the other GPU, or -Force only if it is yours." -ForegroundColor Red
    if ($Go) { throw "refusing to launch onto a busy GPU (use -Force to override)" }
}

# A deck copy carries the chosen GPU so lpopt.inp is never edited.
$deck = 'lpopt.inp'
if ($pick -ne '0') {
    $deck = Join-Path $env:TEMP "lpopt_gpu$pick.inp"
    $text = (Get-Content -LiteralPath 'lpopt.inp' -Raw) `
        -replace '(?m)^gpu\s*=\s*\d+', "gpu = $pick"
    # MUST be BOM-less.  Windows PowerShell 5.1's `Set-Content -Encoding utf8`
    # writes a UTF-8 BOM, and Python's tomllib rejects the leading U+FEFF with
    # "Invalid statement" -- which is exactly how the first GPU 1 launch failed.
    # UTF8Encoding($false) is the only reliable BOM-less writer on 5.1.
    [System.IO.File]::WriteAllText(
        $deck, $text, (New-Object System.Text.UTF8Encoding($false)))
    $head = [System.IO.File]::ReadAllBytes($deck)[0..2] -join ','
    if ($head -eq '239,187,191') { throw "BOM written to $deck despite BOM-less writer" }
    Write-Host "wrote temporary deck $deck (gpu = $pick, BOM-less verified)" -ForegroundColor Yellow
}

# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host '=== arm commands ===' -ForegroundColor Cyan
foreach ($a in $Arm) {
    if (-not $Arms.Contains($a)) { throw "unknown arm '$a'" }
    $targs = $Common + $Arms[$a]
    Write-Host ""
    Write-Host "[$a] $($Notes[$a])" -ForegroundColor Green
    Write-Host "  python -m lpopt.remote --input $deck train -- $($targs -join ' ')"
}

if (-not $Go) {
    Write-Host ""
    Write-Host 'DRY RUN — nothing launched.  Add -Go to launch.' -ForegroundColor Yellow
    Write-Host 'If the working tree changed since the last push, add -Push (or run'
    Write-Host "  python -m lpopt.remote --input $deck push"
    Write-Host 'first).  A push already done for this tree does NOT need repeating.'
    return
}

# -Push is opt-in: a push already done for this working tree need not be repeated.
if ($Push) { Write-Host '=== pushing working tree ==='; Invoke-Remote @('--input',$deck,'push') }

$launched = @()
foreach ($a in $Arm) {
    $targs = $Common + $Arms[$a]
    Write-Host ""
    Write-Host "=== launching $a ===" -ForegroundColor Green
    # ORDER IS LOAD-BEARING.  lpopt.remote's parser has two positionals (``cmd``
    # then ``train_args`` with nargs="*").  argparse consumes positionals greedily
    # at the FIRST opportunity, so an optional placed BETWEEN them
    # (``train --input X -- ...``) makes it match both against the single slot
    # before ``--input``, leaving train_args empty and every forwarded flag an
    # "unrecognized argument".  Wrapper options MUST precede the sub-command.
    Invoke-Remote (@('--input',$deck,'train','--') + $targs)
    $launched += $a
    # Arms are launched one at a time on purpose: they share a GPU and the
    # design's timing estimates assume a single 5-member chunk resident at once.
    if ($a -ne $Arm[-1]) {
        Write-Host "Wait for $a to reach DONE (python -m lpopt.remote --ts <ts> status) before the next arm." -ForegroundColor Yellow
        break
    }
}
Write-Host ""
Write-Host "launched: $($launched -join ', ')" -ForegroundColor Green
Write-Host 'Check with: python -m lpopt.remote --ts <ts> status'
Write-Host 'Pull with:  python -m lpopt.remote --ts <ts> pull'
