@echo off
setlocal
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
cd /d "C:\Users\USER\lpopt_work\kit_frontier"
"C:\Users\USER\lpopt_work\kit_frontier\venv\Scripts\python.exe" -u -m lpopt frontier-produce --input fp_181.inp --round-budget 50 --run-root runs/flat_power_181 --exclude-cells "E1_E2_f113,E1_E2_f117,E1_E2_f121,E1_E2_f125,J1_J2_f113,J1_J2_f117,J1_J2_f121,J1_J2_f125,K1_K2_f113,K1_K2_f117,K1_K2_f121,K1_K2_f125,G3_G4_f113,G3_G4_f117,G3_G4_f121,G3_G4_f125" > fp_181_out.log 2>&1
echo %ERRORLEVEL% > fp_181_rc.txt
endlocal
