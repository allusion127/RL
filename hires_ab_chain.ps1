# hires_ab_chain.ps1 -- run the remaining hires A/B arms sequentially on GPU 0.
# B1 is already launched by hand; this waits for it, then fires A2, A1, A3, A4,
# A6, A5 one at a time (each waits for the previous DONE marker).
# Co-residency with the other user's job is accepted: VRAM 4.5/97 GB free and
# retrain #4 measured no meaningful slowdown under the same condition.
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot
$log = Join-Path $PSScriptRoot 'hires_ab_chain.log'
function Say($m) { "$(Get-Date -Format 'HH:mm:ss')  $m" | Out-File -FilePath $log -Append -Encoding utf8 }

Say "chain start"
$arms = @('A2','A1','A3','A4','A6','A5')

# wait for whatever is currently training to finish (B1)
function Wait-Idle {
    while ($true) {
        $out = & ssh -p 8022 USER@HOST_238 "ps -u USER -o args= | grep -c '[l]popt.model.train'" 2>$null
        $n = 0; [int]::TryParse(($out | Select-Object -First 1), [ref]$n) | Out-Null
        if ($n -eq 0) { return }
        Start-Sleep -Seconds 120
    }
}

Wait-Idle
Say "B1 finished"

foreach ($a in $arms) {
    Say "launching $a"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'launch_hires_ab.ps1') `
        -Arm $a -Gpu 0 -Go -Force *>&1 | Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds 90
    Wait-Idle
    Say "$a finished"
}
Say "chain done"
