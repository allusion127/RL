"""Spot campaign: cell 5.5-5.75_f117, 16 live calls, champion v5_distill_w160.
Reuses the curriculum gate mini-campaign harness.

HISTORICAL NOTE (2026-07-26): this script was written when that harness ranked by
``score_user_criteria`` at ``f_r_limit = gate_min_f_r``.  The harness now ranks by
the flat_power objective (node_peak primary + map_cov secondary) with F_r as the
D1 safety gate only, so a RE-RUN produces a flatness spot campaign, not the
user-criteria one whose result JSON sits next to this file.  ``gate_cyclen_target``
below is now record-only.  Spends real MASTER calls."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(".").resolve()))
from lpopt.config import load_config
from lpopt.curriculum import CurriculumDriver

cfg = load_config(pathlib.Path("lpopt.inp"))
drv = CurriculumDriver(cfg, progress=False)
drv._load_state()
drv.curr.gate_cyclen_target = 625.0
cid = "5.5-5.75_f109"
champ = json.loads(pathlib.Path("data/curriculum/state.json").read_text(encoding="utf-8"))["champion_model_dir"]
print("champion:", champ)
res = drv._gate_mini_campaign(cid, champ, drv.cells_dir / cid)
out = pathlib.Path("data/reports/spotcheck_625_f109_result.json")
out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=1))
for r in res.get("results", []):
    print(r)
