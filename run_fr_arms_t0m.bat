@echo off
REM Re-run of the tier-0 ga80 arms to capture the FLATNESS scalars.
REM The first sweep wrote node_peak = NaN because fr_arms.py took a bare np.max
REM over a plane whose off-slot cells are NaN; the verifier then purged the case
REM outputs, so the maps could not be recovered post hoc.  fr_arms.py now uses
REM lpopt.data.flatness (the same definition the store and the promotion gate
REM use) and saves each plane as map_<arm>.npy.
REM
REM Re-running is sound because MASTER is deterministic here: the A0 replicate
REM reproduced F_r/CBC/cyclen/n_cycles to every printed digit.
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_arms.py --arm A0 --arm A1 --arm A2 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_arms_t0m --workers 12 --max-cycles 16 > runs_fr_arms_t0m.log 2>&1
echo %ERRORLEVEL% > runs_fr_arms_t0m_rc.txt
endlocal
