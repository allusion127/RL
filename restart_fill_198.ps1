# restart_fill_198.ps1 -- runs ON 198.
# The fill started BEFORE the high-resolution harvest code landed, so its labels
# carry maps but no __traj/__axial (NOT retroactive).  This stops it, verifies the
# new code is present, and relaunches; `produce` resumes from the ledger, so the
# already-converged rows are kept and only the in-flight wave is redone.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

# 1. verify the NEW code is actually on disk before killing anything
$edit5 = Join-Path $k 'lpopt\data\edit5.py'
$hasNew = (Select-String -Path $edit5 -Pattern 'def stack_axial' -Quiet) -and
          (Select-String -Path $edit5 -Pattern 'def stack_step_maps' -Quiet)
if (-not $hasNew) {
  Write-Output 'RESTART198 ABORTED: new harvest code NOT found on 198 - nothing stopped.'
  exit 1
}

# 2. stop the running fill (python orchestrator + its MASTER children)
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'fill_198\.inp' })
foreach ($p in $py) { taskkill /F /PID $p.ProcessId /T 2>&1 | Out-Null }
Start-Sleep -Seconds 3
taskkill /F /IM master4.0m4_r1.exe 2>&1 | Out-Null   # 198 runs only our jobs
Start-Sleep -Seconds 3

# 3. relaunch via the existing task (run_fill_198.bat; fp_198_rc.txt already exists
#    so it goes straight to produce)
try { schtasks.exe /delete /tn fill_198 /f | Out-Null } catch {}
& schtasks.exe /create /tn fill_198 /tr "$k\run_fill_198.bat" /sc once /st 23:59 /f | Out-Null
$cc = $LASTEXITCODE
& schtasks.exe /run /tn fill_198 | Out-Null
$rc = $LASTEXITCODE
Start-Sleep -Seconds 20
$n  = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match 'fill_198\.inp' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
Write-Output ("RESTART198 code_ok=True killed=" + $py.Count + " create_rc=" + $cc +
              " run_rc=" + $rc + " fill_python=" + $n + " master=" + $mc)
