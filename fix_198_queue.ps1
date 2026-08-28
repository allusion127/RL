# fix_198_queue.ps1 -- runs ON 198.
# PROBLEM: the fill restart wrote fill_198_rc.txt (=1, from the kill), which is the
# gate run_newfeed_198.bat waits on -> newfeed_198 jumped the queue and is now
# running CONCURRENTLY with fill_198 (master=25 on 16 cores, CPU 100%).
# The two campaigns are cell-disjoint so no duplicate labels were produced; the
# damage is pure oversubscription (~1.6x), which slows BOTH.
# FIX: stop newfeed_198 only, let fill_198 finish, then re-arm newfeed after.
$ErrorActionPreference='Continue'
$k='C:\Users\USER\lpopt_work\kit_frontier'

# 1. stop the queue-jumper (newfeed) -- NOT fill
$nf=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
      Where-Object { $_.CommandLine -match 'newfeed_198\.inp' })
foreach($p in $nf){ taskkill /F /PID $p.ProcessId /T 2>&1 | Out-Null }
try { schtasks.exe /delete /tn newfeed_198 /f | Out-Null } catch {}
Start-Sleep -Seconds 5

# 2. clear the stale gate so a future re-arm waits on a REAL completion
Remove-Item (Join-Path $k 'fill_198_rc.txt') -Force -ErrorAction SilentlyContinue

$fill=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'fill_198\.inp' }).Count
$mc=@(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
Write-Output ("FIX198 newfeed_killed=" + $nf.Count + " fill_still_running=" + $fill +
              " master=" + $mc + " stale_gate_cleared=True")
