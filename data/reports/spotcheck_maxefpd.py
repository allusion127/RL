"""Max-EFPD / min-F_r Pareto spot campaign under constraints (F_q<=2.41,
CBC<=1550, |AO|<=0.30). Cell 5.5-5.75_f141, 16 verified calls."""
import json, pathlib, random, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(".").resolve()))
from lpopt.config import load_config
from lpopt.curriculum import CurriculumDriver, _gen_candidates, _build_resolver, _short_run_root
from lpopt.vendor.masterrl.domain import CaseKey

CID = "5.5-5.75_f141"
RZ = 0.25  # conservatism for constraint screening / cyclen LCB

cfg = load_config(pathlib.Path("lpopt.inp"))
drv = CurriculumDriver(cfg, progress=False)
drv._load_state()
champ = json.loads(pathlib.Path("data/curriculum/state.json").read_text(encoding="utf-8"))["champion_model_dir"]
cell = drv.state["cells"][CID]
band, feed, pairs = cell["band"], cell["feed"], cell["pairs"]
model = drv.load_model(champ)
rng = random.Random(4242)
pool = _gen_candidates(pairs, feed, 640, rng)
pats = [p for _pr, p in pool]; cases = [CaseKey(pr, int(feed)) for pr, _p in pool]
pred = model.predict(pats, cases)
mu = np.asarray(pred.mean); sd = np.asarray(getattr(pred, "std", None) if getattr(pred, "std", None) is not None else np.zeros_like(mu))
FR, CBC, FQ, CY, AO = 0, 1, 2, 3, 4
feas = ((mu[:,FQ]+RZ*sd[:,FQ] <= 2.41) & (mu[:,CBC]+RZ*sd[:,CBC] <= 1550.0)
        & (mu[:,AO]+RZ*sd[:,AO] <= 0.30))
cy_lcb = mu[:,CY] - RZ*sd[:,CY]
idx = np.where(feas)[0]
print(f"pool 640 -> predicted-feasible {len(idx)}")
order_cy = idx[np.argsort(-cy_lcb[idx])]
picks, seen = [], set()
for i in order_cy[:40]:
    k = f"{pool[i][0]}|{pool[i][1].canonical()}"
    if k in seen: continue
    seen.add(k); picks.append(int(i))
    if len(picks) >= 10: break
top40 = [i for i in order_cy[:40] if int(i) not in picks]
for i in sorted(top40, key=lambda j: mu[j,FR]):
    k = f"{pool[i][0]}|{pool[i][1].canonical()}"
    if k in seen: continue
    seen.add(k); picks.append(int(i))
    if len(picks) >= 16: break
pick_cands = [pool[i] for i in picks]
print("verifying", len(pick_cands), "candidates...")
lib = drv.fuel_library()
cell_lib = cell.get("library_id") or drv._band_library(band)
resolver = _build_resolver(cfg, lib, cell_lib)
verifier = drv.verifier(_short_run_root(cfg, CID, "x"), resolver)
entries = drv._wave_entries(pick_cands, feed, resolver, lib, cell_lib)
outcomes = verifier.evaluate_wave(entries)
res = []
for j, (pair, pat) in enumerate(pick_cands):
    oc = outcomes[j]
    row = {"pair": pair, "status": oc.status,
           "pred_cyclen": float(mu[picks[j],CY]), "pred_f_r": float(mu[picks[j],FR])}
    if oc.status == "converged" and oc.fom is not None:
        f = oc.fom
        row.update(cyclen=f.cyclen, f_r=f.f_r, f_q=f.f_q, cbc_max=f.cbc_max, ao_abs=f.ao_abs,
                   feasible=bool(f.f_q<=2.41 and f.cbc_max<=1550 and f.ao_abs<=0.30))
    res.append(row)
out = {"cell": CID, "objective": "max cyclen + min f_r s.t. F_q/CBC/AO", "results": res}
pathlib.Path("data/reports/spotcheck_maxefpd_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
conv = [r for r in res if r["status"]=="converged"]
feas_rows = [r for r in conv if r.get("feasible")]
print(f"converged {len(conv)}/{len(res)}, constraint-feasible {len(feas_rows)}")
for r in sorted(conv, key=lambda r: -(r.get("cyclen") or 0)):
    print({k: round(v,3) if isinstance(v,float) else v for k,v in r.items() if k!="pair"})
