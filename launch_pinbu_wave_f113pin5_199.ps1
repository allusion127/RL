# ad hoc launcher for the 5-core N1_N2/f113 measured-pin re-verification.
# Mirrors launch_pinbu_wave_199.ps1's busy gate + WMI detached-launch mechanism,
# but targets the smaller pinbu_wave_f113pin5 plan/run-dir so it never touches
# or resumes against the 44-chain precedent's runs/pinbu_wave ledger.
$ErrorActionPreference = 'Continue'
$k = 'C:\Users\USER\lpopt_work\kit_frontier'

$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object { $_.CommandLine -match 'lpopt|ablation|batchswap|pinbu' }).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {
  Write-Output ("F113PIN5 REFUSED: box busy (python=$busy master=$mc)"); exit 1
}

$plan = Join-Path $k 'data\reports\pinbu_wave_f113pin5_prereg_20260820.json'
$wantHash = '90E5B94F91185B6D5981B41285EBF1262177E747926EEE747F356511E40156C1'
if (-not (Test-Path $plan)) { Write-Output 'F113PIN5 REFUSED: plan not found'; exit 1 }
$gotHash = (Get-FileHash -Algorithm SHA256 $plan).Hash
if ($gotHash -ne $wantHash) {
  Write-Output "F113PIN5 REFUSED: plan sha256 mismatch (got $gotHash want $wantHash)"; exit 1
}
$deckHash = (Get-FileHash -Algorithm SHA256 (Join-Path $k 'pinbu_wave_199.inp')).Hash
if ($deckHash -ne '03EEADE4AF84DF8C74BF00513869D84BB34A3EA21DDC43F10E6783A0D1541DDA') {
  Write-Output 'F113PIN5 REFUSED: deck sha256 mismatch'; exit 1
}
$harnessHash = (Get-FileHash -Algorithm SHA256 (Join-Path $k 'pinbu_wave.py')).Hash
if ($harnessHash -ne 'CAF07B25E291BE6A6B52657981D05560D2DBD78A403B94B31F3A8D4D166FFD1D') {
  Write-Output 'F113PIN5 REFUSED: harness sha256 mismatch'; exit 1
}

$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB,1)
$diskGB = [math]::Round((Get-PSDrive C).Free/1GB,1)
if ($freeGB -lt 20) { Write-Output "F113PIN5 REFUSED: only $freeGB GB RAM free"; exit 1 }
if ($diskGB -lt 10) { Write-Output "F113PIN5 REFUSED: only $diskGB GB disk free"; exit 1 }

# fresh, dedicated run dir -- 5 targets only, never mixed with runs/pinbu_wave
Remove-Item (Join-Path $k 'runs\pinbu_wave_f113pin5') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_f113pin5_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k 'pinbu_wave_f113pin5_out.log') -Force -EA SilentlyContinue

$py = 'C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe'
$bat = Join-Path $k 'run_pinbu_wave_f113pin5_199.bat'
@"
@echo off
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "$k"
"$py" -u pinbu_wave.py run ^
  --plan data/reports/pinbu_wave_f113pin5_prereg_20260820.json ^
  --deck pinbu_wave_199.inp ^
  --run-dir runs/pinbu_wave_f113pin5 ^
  > pinbu_wave_f113pin5_out.log 2>&1
echo %ERRORLEVEL% > pinbu_wave_f113pin5_rc.txt
endlocal
"@ | Set-Content -Path $bat -Encoding ASCII

$cmdline = '"C:\Windows\System32\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }
Write-Output ("F113PIN5 Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) { Write-Output 'F113PIN5 LAUNCH FAILED'; exit 1 }

Start-Sleep -Seconds 30
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -match 'pinbu_wave' }).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k 'pinbu_wave_f113pin5_out.log'
Write-Output ("F113PIN5 armed: hashes_ok=True python=$n master=$mc2 freeRAM=${freeGB}GB log=$log")
if (Test-Path $log) {
  Write-Output '--- log header ---'
  Get-Content $log -TotalCount 40
} else {
  Write-Output '--- log not created yet ---'
}
