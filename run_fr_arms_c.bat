@echo off
REM TIER 0b: six MORE ga80 dose points for the flat-assembly -> core F_r experiment,
REM all inside the SAME 3_GA_Surrogate/FEASIBLE_PACKAGE as A0/A1/A2, so there is no
REM cross-library confound.  Same fixed loading pattern (flattest feasible core
REM b0ff11ef16de, feed 121); only the two fresh batch identities change.
REM All six resolve at fallback_level 0 with NATIVE restarts (verified 2026-08-09).
REM
REM FF = DeCART pin-power form factor at BU=0, read from the packaged HGC DIST
REM map (the correct burnup index for a FRESH slot).  Predicted F_r uses the
REM TASK-C decomposition F_r = 1.0928 * max(p_boc * FF) with p_boc 1.1934 in the
REM E1-role (68 fresh slots) and 1.2080 in the E2-role (53 fresh slots, hot slot).
REM
REM   C1 H3_H4  FF 1.117/1.171 -> 1.5458  fills the 1.152-1.178 FF_hot gap
REM   C2 K5_K6  FF 1.118/1.149 -> 1.5168  \  FF-matched, reactivity-varying triplet:
REM   C3 L3_L4  FF 1.115/1.145 -> 1.5115   > CBC 1298/1351/1222, cyclen 654/667/639
REM   C4 J1_J2  FF 1.113/1.146 -> 1.5128  /  (spread here = the reactivity confound)
REM   C5 E3_E4  FF 1.101/1.101 -> 1.4534  flat extreme (E3 in ALL 121 fresh slots)
REM   C6 A8_A2  FF 1.178/1.178 -> 1.5551  peaky extreme (A2 in ALL 121 fresh slots)
REM
REM C5/C6 load a SINGLE type into every fresh slot.  They are legal
REM (validate_case only requires fresh batches to be a SUBSET of the pair) and they
REM buy the full ga80 FF_hot span 1.101-1.178, but they also remove the control's
REM high-k/low-k arrangement, so node_peak may move: read C5 vs C6 as their own
REM 2-point slope, NOT against C1-C4 or A0-A2.
REM
REM chcp 65001 + PYTHONIOENCODING: a cp949 console raises UnicodeEncodeError inside
REM the verifier's log path (observed 2026-08-01, exit 90).
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u fr_arms.py --arm C1 --arm C2 --arm C3 --arm C4 --arm C5 --arm C6 ^
  --package ../3_GA_Surrogate/FEASIBLE_PACKAGE ^
  --exe D:/DeCART_MASTER/BIN/master4.0m4_r1.exe ^
  --run-dir runs/fr_arms_c --workers 16 --max-cycles 16 > runs_fr_arms_c.log 2>&1
echo %ERRORLEVEL% > runs_fr_arms_c_rc.txt
endlocal
