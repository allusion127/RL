"""Generate a P3b anchor campaign kit (deck + bat + launcher + status probe).

Three (at most) 3-fresh-type campaigns are launched at the cells the
pre-registered anchor rule picked.  They differ ONLY in cell, feed, case id,
donor pair and the numeric marks; every other knob is the house pattern frozen
by ``fpcamp_minfr_TRIPLE_f125_199.inp`` — the first 3-type campaign, which ran
rc=0, 60/60 calls, 100 % level-3 restart resolution.  Writing them from one
generator rather than by hand is the point: a hand-edited fourth copy of a
200-line deck is how a knob drifts between arms and turns a comparison into a
confound.

Every deck this writes is byte-derived from the same template with the same
substitutions, so ``diff`` across the arms shows exactly the intended fields.

    python make_anchor_campaign.py --from data/reports/.../anchor_selected.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "reports" / "mesh_multitype_20260818"
KIT = "C:\\Users\\USER\\lpopt_work\\kit_frontier"

DECK = '''\
################################################################################
# lpopt FRONTIER-KIT deck — P3b MULTI-TYPE ANCHOR CAMPAIGN {idx}/{ntot}
# {case}  /  feed {feed}  /  {lib}
# champion s1i (cond_schema v8)  |  BOX 199
#   `python -m lpopt optimize --input {deck_name} \\
#        --run-dir runs/{run_name} --no-early-stop`
#
# Written 2026-08-18 from make_anchor_campaign.py.  The cell was chosen by the
# PRE-REGISTERED rule in
#   data/reports/mesh_multitype_20260818/PREREG_multitype_mesh_20260818.md §4.2
# executed mechanically by anchor_select_multitype.py; the selection inputs and
# every rejected cell's reason are in anchor_verdicts.csv.  Nothing in this
# header was written after seeing a MASTER result at this cell.
#
# House pattern: fpcamp_minfr_TRIPLE_f125_199.inp (the programme's first 3-type
# campaign — rc=0, 60/60 calls, 100% level-3 pair_ecore restart resolution).
# Only the cell, feed, case id, donor and the numeric marks differ.
#
################################################################################
# WHY THIS CELL
################################################################################
# The multi-type sweep (P3a, model-only, no MASTER) scored every one of the 90
# mesh cells at 2, 3 and 4 fresh types under the same 5-axis gate, and this cell
# carries one of the largest PREDICTED grading gains that also survives all five
# pre-registered feasibility filters:
#
#   predicted joint-clean F_r floor   2-type  {fr2:.4f}
#                                     3-type  {fr3:.4f}
#   predicted delta(3-2)                      {gain:+.4f}
#   R1 spec                           mono-spec ({anchor_spec})
#   nestedness                        the triple CONTAINS both pair members
#   restart asset                     {n_feed} store rows at this pair+feed
#                                     ({n_any} at any feed)
#
# The 3-type ladder is the cell's own 2-type pair with ONE rung inserted:
{ladder_block}#
# equal-thirds e_core {e3:.6f}   (2-type 50/50 reference: {e2:.6f})
#
################################################################################
# THE MARKS — PINNED BEFORE LAUNCH
################################################################################
#   PRIMARY   a MASTER-verified core passing ALL FIVE gates (F_r<=1.55 &
#             CBC<=1600 & F_q<=2.41 & |AO|<=0.30 & predicted pin<=78).
#   STRETCH   {stretch_txt}
#   NULL      the 3-type F_r floor does not beat {stretch_mark}.  Registered
#             reading: "a third fresh type does not buy F_r at this cell."
#             Reported plainly — it is a result about grading, not a failure.
#
#   CALIBRATION: the sweep's one falsifiable check is the hgd569/f125 cell,
#   where a 3-type delta has actually been measured (-0.0364 joint-clean).
#   calibration_hgd569.json records whether the sweep reproduces that sign.
#   Read it BEFORE reading this campaign's delta as a validation of the sweep.
#
################################################################################
# MODEL CAVEAT — s1i has 39 three-type training rows and ZERO four-type rows
################################################################################
# s1i (cond_v8) is the first promoted checkpoint that can encode a 3..5-type
# core.  Its composition block was trained on 74,537 2-type rows and 39 3-type
# rows (10 more went to val).  `g_type_frac_4/5` never moved at train time
# (ab2_addendum_S1I_20260817.md §3.2).  So wave-0 rankings here are OUT OF
# DISTRIBUTION, and the load-bearing early check is MASTER truth: do the
# graded_morph seeds beat their own 2-type parents in waves 0-1?
#
# This cell has {n_case_rows} store rows under this CASE ID, so the wave
# fine-tune gate {gate_txt}.
#
################################################################################
# FLEET RULES.  199 only (busy gate refuses rather than stacks; deck-hash gate
# refuses on any edit).  198 / 181 / 238 untouched.  Fresh run dir.
# Ship-don't-remote-edit: every file scp'd in whole.
# NOTE (D) — the `[optimize][DEPRECATED]` banner for min_fr_max_cycle is
# EXPECTED (campaign.py), not a failed launch.
# NOTE (R) — `lpopt report`'s "Best verified loading patterns" table ranks by
# CYCLE DISTANCE regardless of objective and will NOT show the F_r winner.
# Read `state.json -> best`.
################################################################################

[flow]
title = "min_fr_max_cycle 3-FRESH-TYPE GRADED — P3b anchor {idx}/{ntot}: {case} / f{feed} ({lib}, e_core {e2:.4f}), champion s1i (v8), CBC gate 1600, pin-BU gate 78"
output_root = "runs"
random_seed = {seed}

[master]
executable = "C:/DeCART_MASTER/BIN/master4.0m4_r1.exe"   # BOX 199
workers = 8
use_all_cores = true
host_reserve = 0
timeout = 3600
max_cycles = 16
consecutive = 2

{routing_block}
[case]
mode = "fixed"
# member order is the radial grading order (hot, mid, cold); `case_batches`
# parses it and `resolved_center` takes the first member.
pair = "{case}"
feed = {feed}

[model]
backend = "posval_cnn"
model_dir = "data/models/s1i"   # 10th champion — cond_v8, the promoted
                                # checkpoint that can encode 3..5 fresh types.
cond_schema = "v8"
map_head_mode = "multiscale"
map_prior_residual = true
map_spectral_weight = 0.3
device = "cpu"
inference = "local_cpu"
library_id = "{lib}"
store_dir = "data/store"

[search]
pool_size = 2500
pool_cap = 100000
elite_frac = 0.65
guided_frac = 0.10
diversity_frac = 0.25
beam_width = 4
completions_per_prefix = 8
n_moves_early = 2
n_moves_late = 5
elite_top_k = 32
# donor = this cell's own 2-type pair.  Without it the elite pool for a graded
# case id is EMPTY and graded_morph is unreachable, and the campaign would NULL
# for a plumbing reason wearing the costume of a physics result.
elite_seed_cases = ["{donor}"]
# every one of the 60 calls must buy a genuinely GRADED core: a board that
# dropped a rung is the same physical core as a 2-type board with an identical
# %LPD_SHF, and re-measuring that population is a wasted MASTER call.
require_all_fresh_types = true
near_miss_f_r = {near_miss:.2f}
near_miss_top_k = 8
dry_run_pool_size = 400

[search.trust_region]
enabled = true
feed_step = 4
e_core_band = 0.10
n_min = 30
promote_after = 16
frontier_sigma_inflation = 1.5
frontier_slots_per_wave = 1

[search.local_search]
top_m = 32
neighbors = 48
depth = 2
max_predictions = 1500
n_moves = 1

[acquisition]
objective = "min_fr_max_cycle"
budget = 60                  # 7 waves x 8 + 4 reserve — the house budget, so
                             # the alphabet stays the only variable across arms.
wave_size = 8
exploit = 5
explore = 2
control = 1
reserve = 4
n_waves = 7
tau0 = 0.30
hamming_min = 4
exploit_verified_hamming = 2
tie_epsilon = 0.1
finetune_new_weight = 4
risk_z = 0.25

minfr_lambda = 1000.0        # "F_r strictly dominates".  lambda = 0 would DELETE
                             # the F_r term and silently invert this into cycle
                             # maximisation.

# --- feasibility: the FIVE-axis gate ----------------------------------------
f_r_limit = 1.55
cbc_limit = 1600.0
f_q_limit = 2.41
ao_abs_limit = 0.30
minfr_pin_bu_limit = 78.0    # LEU+ limit 80 minus 2.0 model margin (pin head
                             # in-cell MAE 1.84, bias -1.39, under-predicting).

# --- REPORT-ONLY cycle readouts.  THEY GATE NOTHING. -------------------------
cycle_target_efpd = {cycle_target:.1f}
cycle_tolerance_efpd = 40.0

# --- wave online-update gate -------------------------------------------------
replay_size = 512
finetune_epochs = 3
holdout_size = 128
gate_epsilon = 0.02
gate_skill_objective = 0.10
gate_skill_halt = 0.0

# --- stopping ----------------------------------------------------------------
min_waves_before_stop = 6
no_improve_waves = 3
dry_run_replay_size = 64
dry_run_finetune_epochs = 1

[produce]
store_dir = "data/store"
ledger = "data/produce/ledger.jsonl"
promoted_root = "data/produce/promoted"
template_fallbacks = []      # {fallback_note}
'''

#: the two production libraries route differently, and mixing them is a silent
#: failure: a ga80 reload template would win resolution against a paramA case
#: and then die on the package's %GEN_DIM sanity gate 40 MASTER-minutes in.
#: Taken verbatim from the decks that already ran on each library —
#: fpcamp_minfr_TRIPLE_f125_199.inp (paramA) and fpcamp_minfr_E1E2_f109_199.inp
#: (ga80) — rather than reasoned about here.
ROUTING = {
    "paramA": ('[verify]\n'
               'package_root = "data/design/package"   # the paramA package '
               '(NOT FEASIBLE_PACKAGE)\n'
               'harvest_maps = true\n\n'
               '[design]\n'
               '# paramA routing (anchors_meshv3_198.inp idiom): a library="paramA"\n'
               '# case goes to THIS package plus the registry alias bridge and the\n'
               "# package's own %GEN_DIM dims.\n"
               'store_dir = "data/design"\n'
               'package_root = "data/design/package"\n'),
    "ga80": ('[verify]\n'
             'package_root = "FEASIBLE_PACKAGE"   # the ga80 package, relative to '
             'the kit root\n'
             'harvest_maps = true\n'),
}
FALLBACK_NOTE = {
    "paramA": "MUST stay empty on paramA: a ga80 reload deck would win resolution "
              "and then fail the paramA %GEN_DIM sanity gate",
    "ga80": "the pair resolves its template via the same-pair tier",
}
#: (package path as the launcher writes it, a file that must exist inside it)
PKG_CHECK = {
    "paramA": ("data\\design\\package", "registry.json"),
    # the ga80 package has no registry.json — its type ids ARE the deck names,
    # so there is no alias bridge to register.  Verified on 199, where the
    # ga80 campaigns that already ran resolve against manifest.csv.
    "ga80": ("FEASIBLE_PACKAGE", "manifest.csv"),
}

BAT = '''\
@echo off
REM P3b MULTI-TYPE ANCHOR CAMPAIGN {idx}/{ntot} -- {case} / feed {feed} / {lib}
REM on BOX 199.  Cell chosen by the pre-registered anchor rule (PREREG §4.2),
REM executed by anchor_select_multitype.py.
REM
REM Objective: minimise F_r under FIVE gates (F_r<=1.55, CBC<=1600, F_q<=2.41,
REM |AO|<=0.30, predicted pin BU<=78).  NO cycle target, NO band, lambda=1000.
REM Budget 60 (7 waves x 8 + 4 reserve) -- the house budget.
REM
REM THE NUMBER TO BEAT: {stretch_mark}.
REM Model: data/models/s1i (cond_v8).
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "{kit}"
set PY=C:\\Users\\USER\\lpopt_work\\kit_pc2\\venv\\Scripts\\python.exe
"%PY%" -u -m lpopt optimize ^
  --input {deck_name} ^
  --run-dir runs/{run_name} ^
  --no-early-stop ^
  > {run_name}_out.log 2>&1
echo %ERRORLEVEL% > {run_name}_rc.txt
endlocal
'''

LAUNCH = '''\
# {launch_name} -- runs ON 199.  P3b multi-type anchor campaign {idx}/{ntot}:
# {case} / feed {feed} / {lib}, champion s1i (v8).
#
# Coordinator runs (this is the ONLY command that starts MASTER):
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File {kit}\\{launch_name}"
#
# BUSY GATE: REFUSES rather than stacks.
# DECK GATE: the deck must be the one the pre-registration hashed.
# MODEL GATE: s1i present AND cond_schema v8 -- a v7/v6b checkpoint would either
#             abort or score the graded core as if a rung were not there.
# STORE GATE: the canonical store carrying the DONOR rows for elite_seed_cases.
$ErrorActionPreference = 'Continue'
$k = '{kit}'

$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object {{ $_.CommandLine -match 'lpopt|ablation|batchswap|mesh' }}).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {{
  Write-Output ("{tag} REFUSED: box busy (python=$busy master=$mc)"); exit 1
}}

$deck = Join-Path $k '{deck_name}'
if (-not (Test-Path $deck)) {{ Write-Output "{tag} REFUSED: deck not found ($deck)"; exit 1 }}
$want = '{deck_sha}'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {{
  Write-Output "{tag} REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}}

$ens = Join-Path $k 'data\\models\\s1i\\ensemble.json'
if (-not (Test-Path $ens)) {{ Write-Output "{tag} REFUSED: data\\models\\s1i not on the kit"; exit 1 }}
if (-not (Test-Path (Join-Path $k 'data\\models\\s1i\\DONE'))) {{
  Write-Output "{tag} REFUSED: data\\models\\s1i carries no DONE marker"; exit 1 }}
$meta = Join-Path $k 'data\\models\\s1i\\member_20260716\\meta.json'
if (-not (Test-Path $meta)) {{ Write-Output "{tag} REFUSED: s1i member meta.json missing"; exit 1 }}
$schema = (Get-Content $meta -Raw | ConvertFrom-Json).cond_schema
if ($schema -ne 'v8') {{
  Write-Output "{tag} REFUSED: s1i cond_schema is '$schema', expected 'v8'"; exit 1 }}

$st = Join-Path $k 'data\\store\\records.parquet'
if (-not (Test-Path $st)) {{ Write-Output "{tag} REFUSED: store missing"; exit 1 }}
$wantStore = '{store_sha}'
$gotStore  = (Get-FileHash -Algorithm SHA256 $st).Hash
if ($gotStore -ne $wantStore) {{
  Write-Output "{tag} REFUSED: store sha256 mismatch (the DONOR rows for elite_seed_cases must be present)"
  Write-Output "  expected $wantStore"
  Write-Output "  found    $gotStore"
  exit 1
}}

$bat = Join-Path $k '{bat_name}'
if (-not (Test-Path $bat)) {{ Write-Output "{tag} REFUSED: run bat not at the kit root"; exit 1 }}

$pkg = Join-Path $k '{pkg_win}'
if (-not (Test-Path (Join-Path $pkg '{pkg_file}'))) {{
  Write-Output "{tag} REFUSED: {pkg_win}\\{pkg_file} missing ({lib} routing)"; exit 1 }}

Remove-Item (Join-Path $k 'runs\\{run_name}') -Recurse -Force -EA SilentlyContinue
Remove-Item (Join-Path $k '{run_name}_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k '{run_name}_out.log') -Force -EA SilentlyContinue

$cmdline = '"C:\\Windows\\System32\\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{
        CommandLine      = $cmdline
        CurrentDirectory = $k
     }}
Write-Output ("{tag} Win32_Process Create: ReturnValue=" + $r.ReturnValue + " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) {{ Write-Output "{tag} LAUNCH FAILED (non-zero ReturnValue)"; exit 1 }}

Start-Sleep -Seconds 90
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object {{ $_.CommandLine -match '{deck_stem}' }}).Count
$mc2 = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$log = Join-Path $k '{run_name}_out.log'
Write-Output ("{tag} armed: deck_sha_ok=True store_sha_ok=True schema=v8 python=$n master=$mc2  log=$log")
if (Test-Path $log) {{ Write-Output '--- log header ---'; Get-Content $log -TotalCount 50 }}
else {{ Write-Output '--- log not created yet ---' }}
'''

STATUS = '''\
# {status_name} -- READ-ONLY status probe.  Starts nothing.
#   ssh USER@HOST_199 "powershell -NoProfile -ExecutionPolicy Bypass -File {kit}\\{status_name}"
$ErrorActionPreference = 'SilentlyContinue'
$k   = '{kit}'
$r   = Join-Path $k 'runs\\{run_name}'
$log = Join-Path $k '{run_name}_out.log'
$rcf = Join-Path $k '{run_name}_rc.txt'

$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object {{ $_.CommandLine -match '{deck_stem}' }}).Count
Write-Output ("PROCS master=$mc python=$py")

if (Test-Path $rcf) {{ Write-Output ("RC " + ((Get-Content $rcf -Raw).Trim())) }}

$st = Join-Path $r 'state.json'
if (Test-Path $st) {{
  Write-Output '<<STATE'
  Get-Content $st -Raw
  Write-Output ''
  Write-Output 'STATE>>'
}}

$lj = Join-Path $r 'labels.jsonl'
if (Test-Path $lj) {{ Write-Output ("NLABELS " + @(Get-Content $lj).Count) }}

if (Test-Path $log) {{
  Write-Output '<<WAVES'
  Select-String -Path $log -Pattern '^\\[optimize\\] wave' | Select-Object -Last 3 | ForEach-Object {{ $_.Line }}
  Write-Output 'WAVES>>'
  Write-Output '<<SEED'
  Select-String -Path $log -Pattern 'elite|store_elite|graded|batches' | Select-Object -First 4 | ForEach-Object {{ $_.Line }}
  Write-Output 'SEED>>'
  Write-Output '<<FAILS'
  Select-String -Path $log -Pattern 'Traceback|CRITICAL|MissingCaseAsset|AssetResolutionError|ModelMismatch|MapHarvestAbort|ERROR|ABORT|HALT|halted|NOT DEFINED IN LPD|neutral' |
    Select-Object -Last 4 | ForEach-Object {{ $_.Line }}
  Write-Output 'FAILS>>'
  Write-Output ("LOGBYTES " + (Get-Item $log).Length)
  Write-Output ("LOGMTIME " + (Get-Item $log).LastWriteTime.ToString('s'))
}}
'''


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=str(OUT / "anchor_selected.csv"))
    ap.add_argument("--nodes", default=str(OUT / "mesh_multitype.csv"))
    args = ap.parse_args()

    sel = pd.read_csv(args.src)
    nodes = pd.read_csv(args.nodes).set_index("cell")
    cases = pd.read_csv(OUT / "case_selection.csv")
    fuel = pd.read_parquet(BASE / "data/store/fuel_types.parquet").set_index(
        ["library_id", "type_id"])
    store = pd.read_parquet(BASE / "data/store/records.parquet")
    store = store[(store.valid == True) & (store.converged == True)]   # noqa: E712
    store_sha = sha256(BASE / "data/store/records.parquet")

    manifest = []
    for idx, r in enumerate(sel.itertuples(), start=1):
        node = nodes.loc[r.cell]
        crow = cases[cases.e_target.round(1) == round(r.e_target, 1)].iloc[0]
        types = str(r.case_3).split("_")
        lib = str(r.library_id)
        tag = f"MT{idx}_e{int(round(r.e_target*10))}_f{int(r.feed)}"
        stem = f"fpcamp_mt{idx}_e{int(round(r.e_target*10))}_f{int(r.feed)}_199"
        deck_name = f"{stem}.inp"
        bat_name = f"run_{stem}.bat"
        run_name = stem.replace("_199", "")
        launch_name = f"launch_{stem}.ps1"
        status_name = f"status_{stem}.ps1"

        # the measured 2-type marks at this exact pair+feed, when they exist
        sub = store[(store.library_id == lib) & (store.case_pair == r.pair)
                    & (store.feed == int(r.feed))]
        clean = sub[(sub.cbc_max <= 1600) & (sub.f_q <= 2.41)
                    & (sub.ao_abs.abs() <= 0.30)]
        measured_clean = float(clean.f_r.min()) if len(clean) else np.nan
        measured_raw = float(sub.f_r.min()) if len(sub) else np.nan
        if np.isfinite(measured_clean):
            stretch_mark = f"{measured_clean:.4f}"
            stretch_txt = (f"F_r < {measured_clean:.4f} CBC/F_q/AO-clean — BEAT THE "
                           f"MEASURED 2-type joint frontier at this exact cell and "
                           f"feed ({len(sub)} converged 2-type rows). THE NUMBER TO BEAT.")
            cycle_target = float(clean.loc[clean.f_r.idxmin(), "cyclen"])
            near_miss = round(min(1.90, measured_clean + 0.07), 2)
        else:
            stretch_mark = f"{float(node['min_f_r_clean_2']):.4f} (PREDICTED)"
            stretch_txt = (f"F_r < {float(node['min_f_r_clean_2']):.4f} — the sweep's "
                           f"PREDICTED 2-type joint-clean floor here.  This cell has NO "
                           f"measured 2-type campaign, so the mark is a prediction and "
                           f"is labelled as one everywhere it appears.")
            cycle_target = float(node.get("clean_cyclen_2", np.nan))
            near_miss = round(min(1.90, float(node["min_f_r_clean_2"]) + 0.07), 2)
        if not np.isfinite(cycle_target):
            cycle_target = 700.0

        ladder = []
        for role, t in zip(("hot", "mid", "cold"), types):
            f = fuel.loc[(lib, t)]
            ladder.append(f"#   {role:5s} {t:<16s} e {float(f.u_avg_enrichment):.4f}  "
                          f"n_gd {int(f.n_gd)}  Gd "
                          f"{'' if not np.isfinite(f.gd_wt) else f'{float(f.gd_wt):.0f}%'}\n")
        spec = fuel.loc[(lib, types[0])]
        anchor_spec = (f"enr_main {float(spec.enr_main):.2f} / enr_zone "
                       f"{float(spec.enr_zone):.4f}"
                       if np.isfinite(spec.enr_main) else
                       f"ga80 family {types[0][0]}")
        n_case_rows = int((store.case_pair == r.case_3).sum())

        sub_map = dict(
            idx=idx, ntot=len(sel), case=r.case_3, feed=int(r.feed), lib=lib,
            deck_name=deck_name, run_name=run_name, bat_name=bat_name,
            launch_name=launch_name, status_name=status_name, tag=tag,
            deck_stem=stem, kit=KIT,
            routing_block=ROUTING[lib], fallback_note=FALLBACK_NOTE[lib],
            pkg_win=PKG_CHECK[lib][0], pkg_file=PKG_CHECK[lib][1],
            donor=r.pair, seed=5700 + idx,
            fr2=float(node["min_f_r_clean_2"]), fr3=float(node["min_f_r_clean_3"]),
            gain=float(node["d_min_f_r_clean_3v2"]),
            e2=float(crow.e_core), e3=float(crow.e_core_3),
            n_feed=int(r.n_store_pair_feed), n_any=int(r.n_store_pair_any),
            ladder_block="".join(ladder), anchor_spec=anchor_spec,
            stretch_mark=stretch_mark, stretch_txt=stretch_txt,
            cycle_target=cycle_target, near_miss=near_miss,
            n_case_rows=n_case_rows,
            gate_txt=("has NO in-cell holdout and will read explore/NaN — it "
                      "cannot veto, and a blind gate must not later be reported "
                      "as a passed gate" if n_case_rows == 0 else
                      f"has {n_case_rows} in-cell rows to hold out"),
            store_sha=store_sha, deck_sha="PLACEHOLDER")

        deck_path = BASE / deck_name
        deck_path.write_text(DECK.format(**sub_map), encoding="utf-8", newline="\r\n")
        sub_map["deck_sha"] = sha256(deck_path)
        (BASE / bat_name).write_text(BAT.format(**sub_map), encoding="utf-8",
                                     newline="\r\n")
        (BASE / launch_name).write_text(LAUNCH.format(**sub_map), encoding="utf-8",
                                        newline="\r\n")
        (BASE / status_name).write_text(STATUS.format(**sub_map), encoding="utf-8",
                                        newline="\r\n")
        manifest.append(dict(idx=idx, cell=r.cell, tag=tag, case=r.case_3,
                             feed=int(r.feed), library_id=lib, donor=r.pair,
                             deck=deck_name, deck_sha256=sub_map["deck_sha"],
                             bat=bat_name, launch=launch_name,
                             status=status_name, run_dir=f"runs/{run_name}",
                             store_sha256=store_sha, stretch_mark=stretch_mark,
                             predicted_fr2=sub_map["fr2"],
                             predicted_fr3=sub_map["fr3"],
                             predicted_gain=sub_map["gain"],
                             measured_2type_clean_f_r=measured_clean,
                             measured_2type_raw_f_r=measured_raw))
        print(f"[{idx}] {r.cell}  {r.case_3} f{r.feed}  deck {deck_name} "
              f"sha {sub_map['deck_sha'][:16]}…")

    json.dump(manifest, open(OUT / "campaign_manifest.json", "w", encoding="utf-8"),
              indent=1, default=float)
    print(f"\nwrote {OUT / 'campaign_manifest.json'}  ({len(manifest)} campaigns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
