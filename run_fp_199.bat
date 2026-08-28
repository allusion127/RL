@echo off
setlocal
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
"C:\Users\USER\lpopt_work\kit_pc2\venv\Scripts\python.exe" -u -m lpopt frontier-produce --input fp_199.inp --round-budget 80 --run-root runs/flat_power_199 --exclude-cells "L1_L2_f113,L1_L2_f117,L1_L2_f121,L1_L2_f125,N1_N2_f113,N1_N2_f117,N1_N2_f121,N1_N2_f125,G3_G4_f113,G3_G4_f117,G3_G4_f121,G3_G4_f125" > fp_199_out.log 2>&1
echo %ERRORLEVEL% > fp_199_rc.txt
endlocal
