"""Offline: extract s1i raw/calibrated F_r + trunk embedding for S1j labelled rows."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"c:/Users/USER/Desktop/CT&RPL/2_Project/KNF_LEU+ 시범연료봉 장전 인허가 과제/2026/2_계산/5_RL")
sys.path.insert(0, str(ROOT))
from lpopt.data.schema import unpack_pattern
from lpopt.model.model_api import PosValCnnBackend, CaseKey
from lpopt.model.cell_calibrate import apply_affine_calibration

OUT = Path(sys.argv[1])
STORE = ROOT / "data/store"
split = json.loads((ROOT / "data/splits/S1j.json").read_text(encoding="utf-8"))
print("split keys:", list(split.keys())[:20])
tr_ids = set(split.get("train_ids") or split.get("train"))
va_ids = set(split.get("val_ids") or split.get("val"))
df = pd.read_parquet(STORE / "records.parquet").drop_duplicates("record_id")
lab = df[(df["converged"] == True) & df["f_xy"].notna()].reset_index(drop=True)  # noqa: E712
fold = np.where(lab["record_id"].isin(tr_ids), "train",
                np.where(lab["record_id"].isin(va_ids), "val", "none"))
lab = lab.assign(_fold=fold)
print(lab["_fold"].value_counts().to_dict())

be = PosValCnnBackend.from_dir(ROOT / "data/models/s1i", store_dir=STORE,
                              library_id="ga80", device="cpu")
print("targets", be.target_names, "fr_cells", len(be._fr_cells),
      "apply_fr", be.apply_fr_calibration, "bin", be.fr_bin_width)

feats: list[torch.Tensor] = []
hooks = []
for m in be.members:
    m.eval()
    hooks.append(m.head_trunk.register_forward_hook(
        lambda mod, i, o: feats.append(o.detach().float().cpu())))

pats = [unpack_pattern(str(p)) for p in lab["pattern"]]
cases = [CaseKey(str(p), int(f)) for p, f in zip(lab["case_pair"], lab["feed"])]
N = len(pats)
CH = 256
mu_all, emb_all = [], []
t0 = time.time()
with torch.no_grad():
    for s in range(0, N, CH):
        e = min(s + CH, N)
        cells_t, g_t = be._encode_batch(pats[s:e], cases[s:e])
        feats.clear()
        mus = []
        for m in be.members:
            out = m(cells_t, g_t)
            mus.append(out["mu"].float().numpy())
        mu_all.append(np.stack(mus))                    # [M, n, T]
        emb_all.append(torch.stack(feats).numpy())      # [M, n, H]
        if s % 1024 == 0:
            print(f"  {e}/{N}  {time.time()-t0:.1f}s", flush=True)
for h in hooks:
    h.remove()
mu_z = np.concatenate(mu_all, axis=1)
emb = np.concatenate(emb_all, axis=1)
print("mu_z", mu_z.shape, "emb", emb.shape, f"{time.time()-t0:.1f}s")

members_raw = mu_z * be.tstd[None, None, :] + be.tmean[None, None, :]
mean_t = members_raw.mean(axis=0)
k_fr = be.target_names.index("f_r")
fr_raw = mean_t[:, k_fr].copy()
keys = be._calib_cell_keys(pats, cases, be.fr_bin_width)
libs = [be.serve_library(p) for p in pats]
fr_cal = apply_affine_calibration(fr_raw, keys, be._fr_cells,
                                  globals_by_lib=be._fr_global, libraries=libs)
np.savez_compressed(
    OUT, fr_raw=fr_raw, fr_cal=fr_cal, emb=emb.mean(axis=0),
    f_xy=lab["f_xy"].to_numpy(float), f_r_meas=lab["f_r"].to_numpy(float),
    node_peak=pd.to_numeric(lab["node_peak"], errors="coerce").to_numpy(float),
    fq=mean_t[:, be.target_names.index("f_q")],
    cyclen=mean_t[:, be.target_names.index("cyclen")],
    cbc=mean_t[:, be.target_names.index("cbc_max")],
    fold=lab["_fold"].to_numpy(), record_id=lab["record_id"].to_numpy(),
    cell=(lab["case_pair"].astype(str) + "/f" + lab["feed"].astype(int).astype(str)).to_numpy(),
)
print("wrote", OUT)
