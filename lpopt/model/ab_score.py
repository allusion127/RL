"""Score trained A/B arms and accumulate the pre-registered results table.

``python -m lpopt.model.ab_score --arm B1=runs/20260725_060441 [...]``

Inference is rebuilt from each member's ``meta.json`` rather than going through
:mod:`.model_api`.  Two reasons, both deliberate:

* the serving path is being edited concurrently (sigma calibration), and scoring
  must not depend on a moving file;
* the serving encoder does not yet read ``meta["power_prior"]``, so it would build
  a cond_v6 arm's ``prior_power`` channel from module DEFAULT constants instead of
  the fitted ones -- i.e. it would score arm A2 on inputs it never trained on.
  Rebuilding here reads the stamped scalars and is correct for every schema.

Featurization dominates the cost (encoding is CPU-bound and single-threaded), so
encoded tensors are cached on disk keyed by (cond_schema, prior constants, row
set).  Arms sharing a schema encode once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..config import FR_GUARD_DEFAULT_DECK, fr_guard_from_deck
from ..data.flatness import map_cov as _map_cov
from ..data.flatness import node_peak as _node_peak
from ..data.fuel_types import FuelLibrary
from ..data.store import StoreReader
from ..vendor.masterrl.domain import SLOTS
from . import ab_eval as M
from . import flat_ab as FA
from .c2_slice import C2Slice, SplitStaleError, build_c2
from .featurize import FeatureEncoder, RecordInputs
from .folds import UNCONTAMINATED_FOLD, FoldFrame, fold_frame, summarize_folds
from .net import PosValNet, PosValNetConfig
from .splits import SplitManifest

_QROW = np.array([s.row for s in SLOTS], dtype=np.intp)
_QCOL = np.array([s.col for s in SLOTS], dtype=np.intp)
#: Map channel carrying BOC assembly power.
_BOC = 0
DEFAULT_CACHE = "runs/_ab_cache"


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@dataclass
class ArmModel:
    """One trained ensemble, rebuilt from its own metadata."""

    model_dir: Path
    members: list[PosValNet]
    meta: dict[str, Any]
    encoder: FeatureEncoder
    device: torch.device

    @classmethod
    def load(cls, model_dir: str | Path, *, device: str = "cpu") -> "ArmModel":
        d = Path(model_dir)
        ens = json.loads((d / "ensemble.json").read_text(encoding="utf-8"))
        member_dirs = [d / m for m in ens["members"]]
        metas = [json.loads((m / "meta.json").read_text(encoding="utf-8"))
                 for m in member_dirs]
        meta = metas[0]
        dev = torch.device(device)

        prior = None
        pp_meta = meta.get("power_prior") or {}
        if pp_meta.get("schema"):
            from .power_prior import PowerPrior
            prior = PowerPrior.from_dict(pp_meta)
        enc = FeatureEncoder(cond_schema=meta["cond_schema"], power_prior=prior)

        members = []
        for mdir, mmeta in zip(member_dirs, metas):
            cfg_kw = {k: v for k, v in mmeta["net_config"].items()
                      if k in PosValNetConfig.__dataclass_fields__}
            net = PosValNet(PosValNetConfig(**cfg_kw))
            net.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu"),
                                strict=True)
            members.append(net.to(dev).eval())
        return cls(model_dir=d, members=members, meta=meta, encoder=enc, device=dev)

    # -- encoding (cached) -------------------------------------------------- #
    def _cache_key(self, record_ids: Sequence[str]) -> str:
        pp = self.meta.get("power_prior") or {}
        sig = json.dumps({
            "schema": self.meta["cond_schema"],
            "m2": pp.get("m2_cm2"), "ex": pp.get("extrap"),
            "n": len(record_ids),
            "ids": hashlib.sha256("".join(record_ids).encode()).hexdigest(),
        }, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:24]

    def encode(self, df: pd.DataFrame, fuel: FuelLibrary, *,
               cache_dir: str | Path | None = DEFAULT_CACHE,
               verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        ids = df["record_id"].astype(str).tolist()
        path = None
        if cache_dir:
            path = Path(cache_dir) / f"enc_{self._cache_key(ids)}.npz"
            if path.exists():
                z = np.load(path)
                return z["cells"], z["globals"]
        t0 = time.time()
        cells, gvecs = [], []
        for _, row in df.iterrows():
            c, g = self.encoder.encode(RecordInputs.coerce(row), fuel)
            cells.append(c)
            gvecs.append(g)
        C = np.stack(cells).astype(np.float32) if cells else np.zeros((0, 1, 19, 19), np.float32)
        G = np.stack(gvecs).astype(np.float32) if gvecs else np.zeros((0, 1), np.float32)
        if verbose:
            print(f"    encoded {len(df)} rows ({self.meta['cond_schema']}) "
                  f"in {time.time() - t0:.1f}s", flush=True)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, cells=C, globals=G)
        return C, G

    # -- forward ------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, cells: np.ndarray, globals_: np.ndarray, *,
                batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """``(scalars[N, T] raw, maps[N, 69] raw boc_power)`` — ensemble means."""
        if len(cells) == 0:
            return np.zeros((0, len(self.meta["target_names"]))), np.zeros((0, 69))
        tz = self.meta["target_zscore"]
        tmean = np.asarray(tz["mean"], dtype=float)
        tstd = np.asarray(tz["std"], dtype=float)
        mz = self.meta["map_zscore"]
        mmean = float(np.asarray(mz["mean"], dtype=float)[_BOC])
        mstd = float(np.asarray(mz["std"], dtype=float)[_BOC])

        mus, maps = [], []
        for s in range(0, len(cells), batch):
            ct = torch.from_numpy(cells[s:s + batch]).to(self.device)
            gt = torch.from_numpy(globals_[s:s + batch]).to(self.device)
            mu_m, map_m = [], []
            for net in self.members:
                out = net(ct, gt)
                mu_m.append(out["mu"].float().cpu().numpy())
                plane = out["map"][:, _BOC].float().cpu().numpy()   # [B, 9, 9]
                map_m.append(plane[:, _QROW, _QCOL])                # [B, 69]
            mus.append(np.mean(mu_m, axis=0))
            maps.append(np.mean(map_m, axis=0))
        mu = np.concatenate(mus) * tstd + tmean
        mp = np.concatenate(maps) * mstd + mmean
        return mu, mp

    def add_cyclen_prior(self, mu: np.ndarray, df: pd.DataFrame,
                         fuel: FuelLibrary) -> np.ndarray:
        """Residual-learned cyclen is returned as prior + residual (absolute)."""
        cp = self.meta.get("cyclen_physics_prior") or {}
        if not cp.get("enabled"):
            return mu
        from .physics_prior import CyclenPhysicsPrior
        prior = CyclenPhysicsPrior(
            alpha=float(cp["alpha"]), beta=float(cp["beta"]),
            rho_leak=float(cp.get("rho_leak", 3500.0)),
            fallback_cyclen=float(cp.get("fallback_cyclen", 0.0)))
        idx = list(self.meta["target_names"]).index("cyclen")
        out = mu.copy()
        out[:, idx] = out[:, idx] + np.asarray(prior.for_rows(df, fuel), dtype=float)
        return out


# --------------------------------------------------------------------------- #
# truth extraction
# --------------------------------------------------------------------------- #
def true_maps(reader: StoreReader, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(mask[N], maps[n_present, 69])`` of actual boc_power at the 69 slots."""
    have, rows = [], []
    keys = df["maps_key"] if "maps_key" in df.columns else pd.Series([None] * len(df))
    for k in keys:
        arr = None if (k is None or (isinstance(k, float) and np.isnan(k))) else reader.maps(str(k))
        if arr is None:
            have.append(False)
            continue
        have.append(True)
        rows.append(np.asarray(arr, dtype=float)[_BOC][_QROW, _QCOL])
    mask = np.asarray(have, dtype=bool)
    return mask, (np.stack(rows) if rows else np.zeros((0, 69)))


def _as_quarter_plane(flat: np.ndarray) -> np.ndarray:
    """``[N, 69]`` -> ``[N, 9, 9]`` with NaN outside the slots (for the FFT)."""
    out = np.full((len(flat), 9, 9), np.nan)
    out[:, _QROW, _QCOL] = flat
    return out


# --------------------------------------------------------------------------- #
# scoring one arm
# --------------------------------------------------------------------------- #
def score_arm(label: str, model_dir: str | Path, *, store_dir: str = M.DEFAULT_STORE,
              splits_dir: str = M.DEFAULT_SPLITS, split: str = M.DEFAULT_SPLIT,
              device: str = "cpu", folds: Sequence[str] = ("C", "B"),
              bootstrap: int = 400, cache_dir: str | None = DEFAULT_CACHE,
              champion_dir: str | None = M.DEFAULT_CHAMPION,
              fr_guarded: bool | None = None,
              deck: str | Path | None = FR_GUARD_DEFAULT_DECK,
              verbose: bool = True) -> dict[str, Any]:
    """Every pre-registered metric for one model dir.

    **The F_r guard is resolved here, not defaulted.**  The fold-C
    ``no_regression_gate`` built below is the FIFTH promotion surface: it is what
    ``ab_decide.evaluate_arms`` consumes as ``passes_gate``, one of the four legs
    of promotion eligibility.  It used to be built with no ``fr_guarded`` and no
    deck, so it resolved to the dataclass default however the deck was set —
    flipping ``[curriculum] gate_noreg_fr_guard_enabled`` would have re-armed
    every other surface and left this one deferred, silently.  It now resolves
    through :func:`..config.fr_guard_from_deck` with the SAME precedence the
    offline flatness A/B uses (explicit ``fr_guarded`` -> ``deck`` -> the field's
    documented default), and the resolution is stamped on the entry as
    ``fr_guard_policy`` so a reader can tell WHICH of the three fired.
    """
    t0 = time.time()
    policy = fr_guard_from_deck(deck, fr_guarded=fr_guarded)
    fr_enforced = bool(policy["enforced"])
    if verbose:
        print(f"[{label}] F_r guard: "
              f"{'ENFORCED' if fr_enforced else 'DEFERRED'} "
              f"(from {policy['source']}; {policy['knob']})", flush=True)
    reader = StoreReader(store_dir)
    fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
    manifest = SplitManifest.from_json(Path(splits_dir) / f"{split}.json")
    df_all = reader.records

    if verbose:
        print(f"[{label}] loading {model_dir}", flush=True)
    arm = ArmModel.load(model_dir, device=device)
    champ = (ArmModel.load(champion_dir, device=device)
             if champion_dir and Path(champion_dir).exists() else None)

    entry: dict[str, Any] = {
        "label": label,
        "model_dir": str(model_dir),
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cond_schema": arm.meta["cond_schema"],
        "net_config": arm.meta["net_config"],
        "n_params_total": arm.meta.get("n_params_total"),
        "n_params_trainable": arm.meta.get("n_params"),
        # WHERE the fold-C gate's F_r setting came from, travelling with the gate
        # it decided -- a consumer must never have to guess which setting the
        # ``pass`` it is about to trust was computed at.
        "fr_guard_policy": policy,
        "power_prior": arm.meta.get("power_prior"),
        "train_config": {k: arm.meta.get("train_config", {}).get(k) for k in
                         ("map_head_mode", "map_prior_residual",
                          "map_spectral_weight", "width", "n_blocks",
                          "head_hidden", "freeze_trunk_cyclen", "init_from")},
        "folds": {},
    }

    for fold in folds:
        ff = fold_frame(df_all, manifest, fold)
        if not len(ff):
            continue
        if verbose:
            print(f"  [{label}] fold {fold} ({ff.name}): {len(ff)} rows, "
                  f"{ff.n_cells} cells", flush=True)
        cells_t, gvecs = arm.encode(ff.df, fuel, cache_dir=cache_dir, verbose=verbose)
        mu, pmap = arm.predict(cells_t, gvecs)
        mu = arm.add_cyclen_prior(mu, ff.df, fuel)
        names = list(arm.meta["target_names"])

        pred: dict[str, np.ndarray] = {n: mu[:, i] for i, n in enumerate(names)}
        truth: dict[str, np.ndarray] = {
            n: pd.to_numeric(ff.df[n], errors="coerce").to_numpy(dtype=float)
            for n in names if n in ff.df.columns}

        # map-derived targets, restricted to rows that actually carry a label
        has_map, tmaps = true_maps(reader, ff.df)
        if has_map.any():
            # ONE definition of both scalars (data.flatness, program §1.1): the
            # multiplicity-weighted CoV.  The local unweighted copy this replaced
            # divided every record by its own 69-slot mean (median 1.0233, range
            # 0.983-1.088), so its numbers were not comparable across records.
            pred["node_peak"] = _node_peak(pmap[has_map])
            pred["map_cov"] = _map_cov(pmap[has_map])
            truth["node_peak"] = _node_peak(tmaps)
            truth["map_cov"] = _map_cov(tmaps)

        fold_out: dict[str, Any] = {
            "name": ff.name, "n": len(ff), "n_cells": ff.n_cells,
            "n_with_map": int(has_map.sum()),
            "n_proposal": int(ff.is_proposal.sum()),
            "uncontaminated": fold == UNCONTAMINATED_FOLD,
            "resolution": {}, "accuracy": {},
        }
        map_cells = ff.cells[has_map] if has_map.any() else np.array([])
        for tgt in M.DELTA_BINS:
            if tgt not in pred or tgt not in truth:
                continue
            c = map_cells if tgt in ("node_peak", "map_cov") else ff.cells
            if not len(c):
                continue
            fold_out["resolution"][tgt] = M.effective_resolution(
                pred[tgt], truth[tgt], c, tgt)
        for tgt in sorted(set(pred) & set(truth)):
            c = map_cells if tgt in ("node_peak", "map_cov") else ff.cells
            if not len(c):
                continue
            fold_out["accuracy"][tgt] = M.within_cell_stats(
                pred[tgt], truth[tgt], c, bootstrap=bootstrap)

        if has_map.any():
            fold_out["map_spectrum"] = M.map_spectrum(
                _as_quarter_plane(pmap[has_map]), _as_quarter_plane(tmaps))
            # post-selection split: model-proposed vs independent production
            prop = ff.is_proposal[has_map]
            if prop.any() and (~prop).any():
                fold_out["map_cov_by_provenance"] = {
                    "proposed": M.within_cell_stats(
                        pred["map_cov"][prop], truth["map_cov"][prop],
                        map_cells[prop], bootstrap=0),
                    "production": M.within_cell_stats(
                        pred["map_cov"][~prop], truth["map_cov"][~prop],
                        map_cells[~prop], bootstrap=0),
                }

        # honest no-regression gate vs the incumbent, on this fold
        if champ is not None and fold == UNCONTAMINATED_FOLD:
            c_cells, c_g = champ.encode(ff.df, fuel, cache_dir=cache_dir,
                                        verbose=verbose)
            c_mu, _ = champ.predict(c_cells, c_g)
            c_mu = champ.add_cyclen_prior(c_mu, ff.df, fuel)
            c_names = list(champ.meta["target_names"])
            old = {n: c_mu[:, i] for i, n in enumerate(c_names)}
            # The ONE switch reaches this gate too (see the docstring): the gate
            # stamps its own ``fr_guard.enforced``, and ab_decide refuses to mix
            # a gate computed at one setting with a judgement made at the other.
            fold_out["gate"] = M.no_regression_gate(
                pred, old, truth, ff.cells, fr_guarded=fr_enforced)
        entry["folds"][fold] = fold_out

    entry["elapsed_s"] = round(time.time() - t0, 1)
    if verbose:
        print(f"[{label}] done in {entry['elapsed_s']}s", flush=True)
    return entry


# --------------------------------------------------------------------------- #
# the flatness A/B: one C2 slice, every arm on the same rows
# --------------------------------------------------------------------------- #
def _arm_predictions(arm: ArmModel, slice_: C2Slice, fuel: FuelLibrary, *,
                     cache_dir: str | None, verbose: bool) -> dict[str, np.ndarray]:
    """One arm's predictions for every judged target on the C2 rows."""
    cells_t, gvecs = arm.encode(slice_.df, fuel, cache_dir=cache_dir,
                                verbose=verbose)
    mu, pmap = arm.predict(cells_t, gvecs)
    mu = arm.add_cyclen_prior(mu, slice_.df, fuel)
    names = list(arm.meta["target_names"])
    out = {n: mu[:, i] for i, n in enumerate(names)}
    # The map-derived scalars come from the SAME canonical definitions the store
    # labelled with (data.flatness), so predicted and stored values are the same
    # quantity rather than two similar-looking ones.
    out["node_peak"] = _node_peak(pmap)
    out["map_cov"] = _map_cov(pmap)
    return out


def score_flatness_ab(arms: dict[str, str], *, control: str,
                      incumbent: str | None = None,
                      store_dir: str = M.DEFAULT_STORE,
                      splits_dir: str = M.DEFAULT_SPLITS,
                      split: str = "S2_flat", device: str = "cpu",
                      cache_dir: str | None = DEFAULT_CACHE,
                      reps: int = 2000, seed: int = 0,
                      allow_stale_split: bool = False,
                      fr_guarded: bool | None = None,
                      deck: str | Path | None = FR_GUARD_DEFAULT_DECK,
                      verbose: bool = True) -> dict[str, Any]:
    """Build the C2 arena, judge every arm against the control, return the slate.

    ``control`` is a required keyword and is looked up in ``arms`` before any
    model is loaded: section 8.4 makes the control mandatory, so failing early
    and loudly beats discovering it after an hour of inference.

    **The F_r guard is resolved here, not defaulted.**  ``fr_guarded`` used to
    exist on this signature and be passed by nobody, and no path read a deck — so
    flipping ``[curriculum] gate_noreg_fr_guard_enabled`` re-armed the curriculum
    gates while this offline A/B stayed silently deferred, which is precisely the
    split-brain the single-switch design exists to prevent.  It now resolves
    through :func:`..config.fr_guard_from_deck`: an explicit ``fr_guarded`` (a CLI
    override or a test) beats ``deck``'s setting, which beats the field's
    documented default.  The resolution — including WHICH of the three fired — is
    stamped on the slate as ``fr_guard_policy``.
    """
    if control not in arms:
        raise FA.ControlMissingError(
            f"control arm {control!r} is not in the arm set {sorted(arms)}.  "
            "Section 8.4: the control is the arm with identical champion init, "
            "identical S2 training set, identical schedule/seed and "
            "map_cov_weight = map_peak_soft_weight = map_cov_rank_weight = 0.  "
            "Without it, arm - incumbent cannot separate the loss effect from "
            "the training-set effect.")

    policy = fr_guard_from_deck(deck, fr_guarded=fr_guarded)
    fr_guarded = bool(policy["enforced"])
    if verbose:
        print(f"[flat-ab] F_r guard: "
              f"{'ENFORCED' if fr_guarded else 'DEFERRED'} "
              f"(from {policy['source']}; {policy['knob']})", flush=True)

    reader = StoreReader(store_dir)
    fuel = FuelLibrary.from_parquet(Path(store_dir) / "fuel_types.parquet")
    manifest = SplitManifest.from_json(Path(splits_dir) / f"{split}.json")
    slice_ = build_c2(reader.records, manifest, allow_stale=allow_stale_split)
    if verbose:
        print(f"[flat-ab] C2: {len(slice_)} rows / {slice_.n_cells} frozen cells "
              f"(fold C was {slice_.provenance['n_fold_c']}; dropped "
              f"{slice_.provenance['dropped']})", flush=True)

    preds: dict[str, dict[str, np.ndarray]] = {}
    loaded: dict[str, ArmModel] = {}
    for label, d in arms.items():
        if verbose:
            print(f"[flat-ab] {label}: {d}", flush=True)
        loaded[label] = ArmModel.load(d, device=device)
        preds[label] = _arm_predictions(loaded[label], slice_, fuel,
                                        cache_dir=cache_dir, verbose=verbose)

    arena = FA.FlatArena.from_c2(slice_, preds, control=control,
                                 incumbent=incumbent)

    # M6: the EXTENDED no-regression gate (map targets included), arm vs the
    # incumbent when there is one -- a per-cell collapse is invisible to any
    # median-over-cells statistic.
    # The gate and the judgement MUST see the same F_r-guard setting, or the
    # judgement's condition-4 branch reports a mismatched input.
    gates: dict[str, dict[str, Any]] = {}
    if incumbent and incumbent in preds:
        truth = {t: slice_.truth(t) for t in M.EXTENDED_GATE_TARGETS}
        for label in arena.challengers:
            gates[label] = M.no_regression_gate(
                preds[label], preds[incumbent], truth, slice_.cells,
                targets=M.EXTENDED_GATE_TARGETS, fr_guarded=fr_guarded)
    reported = {a: FA.reported_effective_resolution(arena, a)
                for a in arena.challengers}
    slate = FA.judge_all(arena, reps=reps, seed=seed, gates=gates,
                         reported=reported, fr_guarded=fr_guarded)
    slate["arm_dirs"] = {k: str(v) for k, v in arms.items()}
    slate["fr_guard_policy"] = policy
    return slate


# --------------------------------------------------------------------------- #
# results accumulation + report
# --------------------------------------------------------------------------- #
def merge_flat_slate(slate: dict[str, Any],
                     path: str | Path = M.DEFAULT_RESULTS) -> dict[str, Any]:
    """Write the slate and each arm's paired block into the results document.

    ``ab_decide`` reads ``arms[<label>]['paired']['C2']`` for its interval
    requirement and ``flat_slate`` for the flatness rule's own verdict; both come
    from here, so the two can never be computed from different runs.
    """
    p = Path(path)
    doc: dict[str, Any] = {"schema": "hires_ab_results_v1", "arms": {}}
    if p.exists():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    doc.setdefault("arms", {})
    for label, judgement in (slate.get("judgements") or {}).items():
        entry = doc["arms"].setdefault(label, {"label": label})
        entry.setdefault("model_dir", (slate.get("arm_dirs") or {}).get(label))
        entry.setdefault("paired", {})["C2"] = FA.paired_block(judgement)
    doc["flat_slate"] = slate
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=False, ensure_ascii=False),
                 encoding="utf-8")
    return doc


def update_results(entry: dict[str, Any], path: str | Path = M.DEFAULT_RESULTS,
                   ) -> dict[str, Any]:
    """Insert/replace one arm's entry, preserving the rest."""
    p = Path(path)
    doc: dict[str, Any] = {"schema": "hires_ab_results_v1", "arms": {}}
    if p.exists():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    doc.setdefault("arms", {})[entry["label"]] = entry
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=False), encoding="utf-8")
    return doc


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render_markdown(doc: dict[str, Any], *, fold: str = UNCONTAMINATED_FOLD) -> str:
    """The arm comparison table (fold C unless asked otherwise)."""
    arms = doc.get("arms", {})
    order = [a for a in ("B0", "B1", "A1", "A2", "A3", "A4", "A5", "A6") if a in arms]
    order += [a for a in sorted(arms) if a not in order]
    L: list[str] = []
    L.append(f"# hires A/B — fold {fold} 결과 (자동 생성)")
    L.append("")
    L.append(f"갱신: {doc.get('updated_at', '—')} · arm {len(order)}개 · "
             f"판정 지표는 `hires_model_ab_design_20260725.md` §6 사전등록판")
    L.append("")
    L.append("## 1차 — 실효 분해능 Δ₇₅/셀내SD (낮을수록 좋음)")
    L.append("")
    L.append("| arm | 파라미터 | node_peak | map_CoV | F_r | cyclen |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for a in order:
        f = arms[a].get("folds", {}).get(fold, {})
        res = f.get("resolution", {})
        cellsn = arms[a].get("n_params_total")
        row = [a, f"{cellsn:,}" if cellsn else "—"]
        for t in M.PRIMARY_TARGETS:
            row.append(_fmt(res.get(t, {}).get("delta75_over_sd"), 2))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## 2차 — fold C 셀내 Spearman ρ (높을수록 좋음)")
    L.append("")
    L.append("| arm | node_peak | map_CoV | F_r | cyclen | 게이트 |")
    L.append("|---|---:|---:|---:|---:|:--:|")
    for a in order:
        f = arms[a].get("folds", {}).get(fold, {})
        acc = f.get("accuracy", {})
        row = [a]
        for t in M.PRIMARY_TARGETS:
            row.append(_fmt(acc.get(t, {}).get("median_rho")))
        g = f.get("gate")
        row.append("—" if not g else ("PASS" if g.get("pass") else
                                      f"FAIL ({g.get('worst_drop', 0):.3f})"))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## 보조 — 축소편향(셀내 예측SD/실측SD, 1.0이 이상) · 맵 나이퀴스트 대역")
    L.append("")
    L.append("| arm | SD비 node_peak | SD비 map_CoV | 나이퀴스트 파워비 | 나이퀴스트 모드 ρ |")
    L.append("|---|---:|---:|---:|---:|")
    for a in order:
        f = arms[a].get("folds", {}).get(fold, {})
        acc = f.get("accuracy", {})
        spec = f.get("map_spectrum") or []
        nyq = spec[-1] if spec else {}
        L.append("| " + " | ".join([
            a,
            _fmt(acc.get("node_peak", {}).get("sd_ratio")),
            _fmt(acc.get("map_cov", {}).get("sd_ratio")),
            _fmt(nyq.get("power_ratio")),
            _fmt(nyq.get("mode_rho")),
        ]) + " |")
    L.append("")
    L.append("> 판정 규칙(사전등록): ① node_peak 또는 map_CoV의 Δ₇₅/SD가 B1 대비 "
             "**≥0.15 개선**되고 타 3타깃 0.05 이상 악화 없음, ② 게이트 PASS, "
             "③ cyclen·F_r 셀내 ρ가 B0 대비 CI 하한 ≥ −0.02.")
    L.append("> A5(순수 용량)가 1차 지표에서 +0.02를 넘지 못하면 "
             "\"단순 폭 확대 무효\"가 세 번째로 확정된다.")
    L.append(">")
    L.append("> **Δ₇₅/SD는 구간 양자화된 값이다.** Δ₇₅는 구간 하한이고 분모(셀내 실측 SD)는 "
             "arm과 무관하므로, 두 arm이 같은 구간을 넘기면 값이 **정확히 같아진다** — "
             "동률은 동등하다는 뜻이 아니다. 동률일 때는 map_CoV·셀내 ρ·SD비·스펙트럼으로 "
             "가려야 한다 (실제로 A1·A4·A6이 node_peak에서 동률이었으나 map_CoV에서 "
             "A4만 미개선이었다).")
    L.append(">")
    L.append("> **위 '게이트' 열은 프록시다** (fold C, ε=0.134). 승격의 권위 게이트는 "
             "`lpopt gate-promote`이며 커리큘럼 `val_by_cell`·`done_cells` 72검사 + "
             "레거시 tail 게이트를 별도로 돌린다. 조회만 하려면 반드시 `--check-only`를 "
             "붙일 것 — 없으면 PASS 즉시 승격된다.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lpopt.model.ab_score")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=DIR",
                    help="arm to score (repeatable), e.g. B1=data/models/2026...")
    ap.add_argument("--store-dir", default=M.DEFAULT_STORE)
    ap.add_argument("--splits-dir", default=M.DEFAULT_SPLITS)
    ap.add_argument("--split", default=M.DEFAULT_SPLIT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="C,B")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--champion", default=M.DEFAULT_CHAMPION,
                    help="incumbent for the no-regression gate ('' to skip)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--results", default=M.DEFAULT_RESULTS)
    ap.add_argument("--markdown", default="data/reports/hires_ab_results.md")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads (production boxes are busy; keep low)")
    ap.add_argument("--folds-summary", action="store_true",
                    help="print the fold provenance table and exit")
    ap.add_argument("--flat-ab", action="store_true",
                    help="run the flatness A/B apparatus (program section 8) on "
                         "the C2 slice instead of the per-arm fold tables")
    ap.add_argument("--control", default="B1",
                    help="the section 8.4 control arm label (mandatory for --flat-ab)")
    ap.add_argument("--incumbent", default="B0",
                    help="incumbent label for the three-way attribution ('' to skip)")
    ap.add_argument("--flat-split", default="S2_flat",
                    help="the flatness program's split (see "
                         "python -m lpopt.tools.audit_c2_split)")
    ap.add_argument("--reps", type=int, default=2000,
                    help="paired bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-stale-split", action="store_true",
                    help="reproduce a historical run on a stale split; the "
                         "artifact still records it as stale")
    ap.add_argument("--deck", default=FR_GUARD_DEFAULT_DECK,
                    help="deck the F_r-guard switch is resolved from "
                         "([curriculum] gate_noreg_fr_guard_enabled), for BOTH "
                         "the per-arm fold gate and --flat-ab; '' to use the "
                         "documented default instead of reading a deck")
    ap.add_argument("--fr-guarded", dest="fr_guarded",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="override the deck's F_r-guard setting for this run "
                         "(applies to both paths)")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, int(args.threads)))

    if args.flat_ab:
        pairs: dict[str, str] = {}
        for spec in args.arm:
            if "=" not in spec:
                ap.error(f"--arm must be LABEL=DIR (got {spec!r})")
            label, d = spec.split("=", 1)
            pairs[label] = d
        try:
            slate = score_flatness_ab(
                pairs, control=args.control,
                incumbent=(args.incumbent or None),
                store_dir=args.store_dir, splits_dir=args.splits_dir,
                split=args.flat_split, device=args.device,
                cache_dir=args.cache_dir or None, reps=args.reps, seed=args.seed,
                allow_stale_split=args.allow_stale_split,
                deck=(args.deck or None), fr_guarded=args.fr_guarded)
        except (FA.ControlMissingError, SplitStaleError) as exc:
            print(f"REFUSED: {exc}")
            return 2
        print("")
        print(FA.render_slate(slate))
        merge_flat_slate(slate, args.results)
        print(f"\n  -> {args.results}", flush=True)
        return 0

    if args.folds_summary:
        reader = StoreReader(args.store_dir)
        manifest = SplitManifest.from_json(Path(args.splits_dir) / f"{args.split}.json")
        print(json.dumps(summarize_folds(reader.records, manifest), indent=1))
        return 0

    doc = None
    for spec in args.arm:
        if "=" not in spec:
            ap.error(f"--arm must be LABEL=DIR (got {spec!r})")
        label, d = spec.split("=", 1)
        entry = score_arm(label, d, store_dir=args.store_dir,
                          splits_dir=args.splits_dir, split=args.split,
                          device=args.device,
                          folds=tuple(f for f in args.folds.split(",") if f),
                          bootstrap=args.bootstrap,
                          cache_dir=args.cache_dir or None,
                          champion_dir=(args.champion or None),
                          deck=(args.deck or None), fr_guarded=args.fr_guarded)
        doc = update_results(entry, args.results)
        print(f"  -> {args.results}", flush=True)

    if doc is None and Path(args.results).exists():
        doc = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if doc and args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(render_markdown(doc), encoding="utf-8")
        print(f"  -> {args.markdown}", flush=True)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
