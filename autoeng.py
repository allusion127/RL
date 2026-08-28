#!/usr/bin/env python
"""autoeng — 자동 엔지니어 (the automatic engineer): cell-sequential auto-open loop.

Point it at a list of ``(pair, feed, library)`` target cells and it runs, per cell,
the exact sequence that a human ran by hand for ``N1_N2/f113`` on 2026-08-16:

    0 PRECHECK  -> 1 PROBE -> 2 OPEN -> 3 HARVEST+MERGE -> 4 RETRAIN+GATE
                -> 5 MAP UPDATE -> 6 NEXT CELL

**Nothing here is a new algorithm.**  Every stage is a *composition* of a piece
that has already been proven in this repository, and the file names its source:

| stage | composed from | precedent |
|---|---|---|
| 0 PRECHECK | ``lpopt.search.resolver.build_case_resolver`` + ``CaseAssetResolver.resolve`` (ladder dry-run), ``data/store/records.parquet``, ``data/reports/dbx_frontier_table.csv``, ``scoping_mesh_20260815/cell_verdicts.csv`` | the f113 deck header's "THE MARKS" block, done by hand |
| 1 PROBE | ``CurriculumDriver._phase_blind_probe`` (plan sec. 12.3) | ``data/curriculum/cells/*/blind_probe.json`` |
| 2 OPEN | ``lpopt optimize`` with the ``fpcamp_minfr_N1N2_f113_199.inp`` knob set carried **verbatim** | ``data/reports/fpcamp_N1N2_f113_results_20260816.md`` — 41 feasible / 100 calls |
| 3 HARVEST+MERGE | ``lpopt merge-store`` (+ the manual pre-merge backup discipline) | same report, sec. 5 |
| 4 RETRAIN+GATE | ``build_split_S1b.py`` -> ``lpopt.remote push/train/status/pull`` -> ``lpopt gate-promote`` | rounds 1-11, ``ab2_addendum_S1G_20260816.md`` |
| 5 MAP UPDATE | ``scoping_mesh.py`` -> ``mesh_vs_db.py`` -> ``--figure-only`` | ``scoping_mesh_20260815/README.md`` |
| 6 ORDER | ``dbx_frontier_note_20260816.md`` sec. 4 screen + the sec. 10.3 transfer measurement | 82-86 % of the neighbour-feed gain was model-attributed |

The driver's own contribution is exactly three things:

1. **Prereg automation.**  Every cell's marks (store floor, DB truth, mesh
   verdict, the model's predicted floor) are measured and *frozen into a
   pre-registration and a deck header BEFORE a single MASTER call is spent*.
   The honesty discipline is the thing most likely to be lost to automation, so
   it is the thing that runs first and blocks the campaign step if it fails.
2. **An append-only state log** so a ``kill -9`` costs at most one step, and
   every MASTER call is attributed to a cell and a stage.
3. **Config-flagged human gates.**  v1 stops for a new assembly, for a failed
   promotion gate, and for a budget overrun.  Everything else is autonomous.

USAGE
    python autoeng.py --config autoeng.toml --dry-run        # plan only, executes nothing
    python autoeng.py --config autoeng.toml                  # live
    python autoeng.py --config autoeng.toml --resume         # after a kill / a gate pause
    python autoeng.py --config autoeng.toml --status         # read the state log

FLEET (standing rules, enforced by ``_fleet_guard``)
    199  campaigns + probe + the model-only mesh sweep
    238  training only (GPU 1 via ``lpopt_gpu1.inp``)
    181  NEVER
    198  production only — autoeng must never address it

WHAT v1 CANNOT DO — stated here so nobody has to infer it:
    * A genuinely NEW assembly (no library entry) is PRECHECKED and then HALTS
      for a human.  The DeCART2D leg is expensive and stays gated.
    * The remote legs (ssh to 199 / 238) are wired but have never been exercised
      by this file.  The first live run must be babysat.
    * The OPEN stage is the f113 recipe; it is one cell's evidence, not a law.
      A cell where it stalls is a result, not a bug — see the NULL branch that
      every generated prereg carries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parent

#: The proven OPEN recipe.  Every knob not listed in ``CELL_OVERRIDE_KEYS`` is
#: carried from this deck byte-for-byte (after a TOML round-trip).
DEFAULT_PARENT_DECK = "fpcamp_minfr_N1N2_f113_199.inp"

#: ``[constraints]`` is INERT on ``min_fr_max_cycle`` and the f113 results report
#: (defect 2) says to drop it from the next min_fr deck.  The automation carries
#: that recorded correction forward instead of re-inheriting the dead block.
DROP_SECTIONS = ("constraints",)

#: Seeds already spent by a campaign in this programme.  A new cell must not
#: reuse one, or wave 0 redraws a pool whose MASTER calls are already bought.
USED_SEEDS = (830, 1103, 1111, 1201, 1600)

#: The s1f/s1g training recipe.  ``ab2_addendum_S1G_20260816.md`` sec. 4: the ONLY
#: per-round delta is ``--split`` and ``--ts``.  Verbatim from data/models/s1g/run.sh.
TRAIN_RECIPE = (
    "--ensemble", "5", "--split", "{split}", "--cond-schema", "v6b",
    "--width", "224", "--n-blocks", "8", "--head-hidden", "384",
    "--epochs", "150", "--num-workers", "8", "--device", "auto",
    "--parallel-members", "5", "--base-seed", "20260716",
    "--map-decoder", "multiscale", "--map-prior-residual",
    "--map-spectral-weight", "0.3", "--map-peak-weight", "2.0",
    "--cyclen-physics-prior", "--quantile-heads", "--quantile-weight", "0.2",
    "--promote-max-asm-bu",
    "--distill-targets", "data/models/_v5_distill_soft.npz",
    "--distill-weight", "0.4", "--distill-min-match-frac", "0.5",
    "--f-r-rank-weight", "0.1",
)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    """One cell to open.  ``new_assembly`` targets carry a spec instead of a pair."""

    pair: str = ""
    feed: int = 0
    library: str = "ga80"
    budget: int = 100                     # OPEN-stage MASTER calls
    probe_budget: int = 8                 # PROBE-stage MASTER calls
    new_assembly: str = ""                # non-empty -> design-chain precheck + HALT
    note: str = ""

    @property
    def cell_id(self) -> str:
        return self.new_assembly or f"{self.pair}_f{self.feed}"

    @property
    def is_new_assembly(self) -> bool:
        return bool(self.new_assembly)


@dataclass
class Fleet:
    campaign_host: str = "USER@HOST_199"
    campaign_kit: str = "C:/Users/USER/lpopt_work/kit_frontier"
    campaign_python: str = "C:/Users/USER/lpopt_work/kit_pc2/venv/Scripts/python.exe"
    train_deck: str = "lpopt_gpu1.inp"     # pins GPU 1 on 238
    #: Boxes autoeng must never address.  181 = never, 198 = production only.
    forbidden: tuple[str, ...] = ("HOST_181", "HOST_198", ".181", ".198")


@dataclass
class AutoengConfig:
    run_id: str = "autoeng"
    root: Path = ROOT
    parent_deck: str = DEFAULT_PARENT_DECK
    main_deck: str = "lpopt.inp"
    champion_state: str = "data/curriculum/state.json"
    store_dir: str = "data/store"
    package_root: str = "../3_GA_Surrogate/FEASIBLE_PACKAGE"
    frontier_table: str = "data/reports/dbx_frontier_table.csv"
    mesh_dir: str = "data/reports/scoping_mesh_20260815"
    notebook: str = "data/reports/AUTOENG_LOG.md"
    probe_budget: int = 8
    open_budget: int = 100
    master_budget_total: int = 500
    pause_for_approval: tuple[str, ...] = (
        "new_assembly", "retrain_promote_fail", "budget_exceeded",
    )
    targets: tuple[Target, ...] = ()
    fleet: Fleet = field(default_factory=Fleet)

    # -- resolved paths ----------------------------------------------------- #
    def p(self, rel: str) -> Path:
        q = Path(rel)
        return q if q.is_absolute() else (self.root / q)

    @property
    def run_dir(self) -> Path:
        return self.root / "data" / "autoeng" / self.run_id

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.jsonl"

    def cell_dir(self, cell_id: str) -> Path:
        return self.run_dir / "cells" / cell_id


_TARGET_KEYS = {"pair", "feed", "library", "budget", "probe_budget", "new_assembly", "note"}
_FLEET_KEYS = {"campaign_host", "campaign_kit", "campaign_python", "train_deck", "forbidden"}
_AE_KEYS = {
    "run_id", "parent_deck", "main_deck", "champion_state", "store_dir", "package_root",
    "frontier_table", "mesh_dir", "notebook", "probe_budget", "open_budget",
    "master_budget_total", "pause_for_approval",
}


def load_autoeng_config(path: str | Path, *, root: Path | None = None) -> AutoengConfig:
    """Read an autoeng TOML.  Unknown keys are a hard error (deck-loader discipline)."""

    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = root or path.resolve().parent

    ae = dict(raw.get("autoeng", {}))
    bad = set(ae) - _AE_KEYS
    if bad:
        raise ValueError(f"{path}: unknown [autoeng] key(s): {sorted(bad)}")

    fl = dict(raw.get("fleet", {}))
    bad = set(fl) - _FLEET_KEYS
    if bad:
        raise ValueError(f"{path}: unknown [fleet] key(s): {sorted(bad)}")
    if "forbidden" in fl:
        fl["forbidden"] = tuple(fl["forbidden"])
    fleet = Fleet(**fl)

    targets: list[Target] = []
    for i, t in enumerate(raw.get("targets", [])):
        bad = set(t) - _TARGET_KEYS
        if bad:
            raise ValueError(f"{path}: unknown [[targets]] key(s) at #{i}: {sorted(bad)}")
        targets.append(Target(**t))

    if "pause_for_approval" in ae:
        ae["pause_for_approval"] = tuple(ae["pause_for_approval"])
    cfg = AutoengConfig(root=root, fleet=fleet, targets=tuple(targets), **ae)

    # Apply the run-level budgets to any target that did not name its own.
    filled = []
    for t in cfg.targets:
        if t.budget == 100 and cfg.open_budget != 100:
            t = Target(**{**t.__dict__, "budget": cfg.open_budget})
        if t.probe_budget == 8 and cfg.probe_budget != 8:
            t = Target(**{**t.__dict__, "probe_budget": cfg.probe_budget})
        filled.append(t)
    cfg.targets = tuple(filled)
    _fleet_guard(cfg)
    return cfg


def _fleet_guard(cfg: AutoengConfig) -> None:
    """Refuse a config that addresses a forbidden box (181 never, 198 production)."""

    surface = " ".join([cfg.fleet.campaign_host, cfg.fleet.campaign_kit,
                        cfg.fleet.campaign_python, cfg.fleet.train_deck])
    for bad in cfg.fleet.forbidden:
        if bad and bad in surface:
            raise ValueError(
                f"[autoeng] FLEET GUARD: config addresses {bad!r}, which autoeng may "
                f"never use (181 = never, 198 = production only)."
            )


def guard_argv(cfg: AutoengConfig, argv: Sequence[str]) -> None:
    """Same guard, applied to every command before it is executed."""

    joined = " ".join(argv)
    for bad in cfg.fleet.forbidden:
        if bad and bad in joined:
            raise RuntimeError(f"[autoeng] FLEET GUARD refused a command touching {bad!r}: {joined}")


# --------------------------------------------------------------------------- #
# append-only state log  (kill -9 resumable; every MASTER call accounted)
# --------------------------------------------------------------------------- #
class StateLog:
    """One JSON object per line, fsync'd, never rewritten.

    A crash can therefore truncate at most the final line, and replaying the file
    reconstructs exactly which steps finished.  ``master_calls`` is carried on the
    ``step_done`` payload so the budget ledger survives the crash too.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._events: list[dict[str, Any]] = []
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        good = 0                        # byte offset of the end of the last valid line
        for line in raw.split(b"\n"):
            text = line.strip()
            if not text:
                good += len(line) + 1
                continue
            try:
                self._events.append(json.loads(text.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                break                   # torn tail from a kill -9
            good += len(line) + 1
        good = min(good, len(raw))
        if good < len(raw):
            # REPAIR the tail.  Without this, the next append would concatenate onto
            # the half-written line and corrupt the record that survived the crash.
            with open(self.path, "r+b") as fh:
                fh.truncate(good)

    # -- write ------------------------------------------------------------- #
    def append(self, kind: str, *, cell: str = "", step: str = "", **payload: Any) -> dict:
        ev = {"seq": len(self._events), "ts": time.time(),
              "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "kind": kind, "cell": cell, "step": step, **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)   # lazily: a dry run
        with open(self.path, "a", encoding="utf-8") as fh:    # must create nothing
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            fh.flush()
        self._events.append(ev)
        return ev

    # -- read -------------------------------------------------------------- #
    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def step_status(self, cell: str, step: str) -> str | None:
        """``done`` / ``skipped`` / ``failed`` / ``started`` / None, last write wins."""
        out = None
        for ev in self._events:
            if ev.get("cell") == cell and ev.get("step") == step:
                out = {"step_start": "started", "step_done": "done",
                       "step_skip": "skipped", "step_fail": "failed"}.get(ev["kind"], out)
        return out

    def master_calls(self) -> int:
        return sum(int(e.get("master_calls", 0)) for e in self._events if e["kind"] == "step_done")

    def result(self, cell: str, step: str) -> dict[str, Any]:
        for ev in reversed(self._events):
            if ev.get("cell") == cell and ev.get("step") == step and ev["kind"] == "step_done":
                return dict(ev.get("result", {}))
        return {}

    def opened_cells(self) -> list[str]:
        return [e["cell"] for e in self._events if e["kind"] == "cell_done" and e.get("cell")]


# --------------------------------------------------------------------------- #
# measurement — THE MARKS (pinned before any MASTER call)
# --------------------------------------------------------------------------- #
def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def derive_seed(pair: str, feed: int, used: Sequence[int] = USED_SEEDS) -> int:
    """Deterministic fresh seed for a cell, never colliding with a spent one."""

    base = 2000 + _stable_hash(f"{pair}_f{feed}") % 800
    while base in set(used):
        base += 1
    return base


def measure_marks(cfg: AutoengConfig, target: Target) -> dict[str, Any]:
    """Measure every mark the prereg pins.  Reads only; writes nothing.

    Sources, all named in the returned dict so a reader can re-derive each number:
      * store floor / support -> ``data/store/records.parquet``
      * DB truth              -> ``data/reports/dbx_frontier_table.csv``
      * model predicted floor + bias + verdict -> ``scoping_mesh_.../cell_verdicts.csv``
    """

    import pandas as pd

    sys.path.insert(0, str(cfg.root))
    pair, feed = target.pair, int(target.feed)
    m: dict[str, Any] = {"pair": pair, "feed": feed, "library": target.library,
                         "sources": {}, "notes": []}

    # ---- store -------------------------------------------------------------
    recs = cfg.p(cfg.store_dir) / "records.parquet"
    m["sources"]["store"] = f"{cfg.store_dir}/records.parquet"
    df = pd.read_parquet(recs)
    lib = df[(df["library_id"] == target.library)]
    cell = lib[(lib["case_pair"] == pair) & (lib["feed"] == feed)]
    conv = cell[cell["converged"].fillna(False).astype(bool) & cell["valid"].fillna(False).astype(bool)]
    m["store_rows"] = int(len(cell))
    m["store_converged"] = int(len(conv))
    m["store_f_r_floor"] = float(conv["f_r"].min()) if len(conv) else None
    m["store_f_r_p10"] = float(conv["f_r"].quantile(0.10)) if len(conv) else None
    m["store_cyclen_min"] = float(conv["cyclen"].min()) if len(conv) else None
    m["store_cyclen_max"] = float(conv["cyclen"].max()) if len(conv) else None
    feas = conv[(conv["f_r"] <= 1.55) & (conv["cbc_max"] <= 1600.0)
                & (conv["f_q"] <= 2.41) & (conv["ao_abs"].abs() <= 0.30)]
    m["store_feasible"] = int(len(feas))
    m["store_convergence_rate"] = round(len(conv) / len(cell), 4) if len(cell) else None
    e_core = lib[lib["case_pair"] == pair]["e_core"]
    m["e_core"] = float(e_core.median()) if len(e_core) else None
    prov = conv["restart_provenance"].value_counts().head(3).to_dict() if len(conv) else {}
    m["store_restart_provenance"] = {str(k): int(v) for k, v in prov.items()}

    # Trust-region support, computed with the ACQUISITION's own bin definition
    # (`TrustRegion.from_store`: exact feed x `_e_bin(e_core, e_core_band)`, over
    # EVERY store row, not just this pair's and not just the converged ones).  The
    # f113 deck header quoted 128 here, which was actually that cell's converged
    # count — a different quantity.  This is the number `n_min` is compared to.
    if m["e_core"] is not None:
        from lpopt.search.acquisition import _e_bin
        want = _e_bin(m["e_core"], 0.10)
        bins = df["e_core"].map(lambda e: _e_bin(e, 0.10))
        m["trust_region_support"] = int(((df["feed"] == feed) & (bins == want)).sum())
    else:
        m["trust_region_support"] = 0

    # cross-feed elite parents: the mechanism the f113 campaign actually ran on.
    pair_rows = lib[lib["case_pair"] == pair]
    pconv = pair_rows[pair_rows["converged"].fillna(False).astype(bool)]
    pfeas = pconv[(pconv["f_r"] <= 1.55) & (pconv["cbc_max"] <= 1600.0)
                  & (pconv["f_q"] <= 2.41) & (pconv["ao_abs"].abs() <= 0.30)]
    m["elite_parents_feasible"] = int(len(pfeas))
    m["elite_parent_feeds"] = {str(int(k)): int(v)
                               for k, v in pfeas["feed"].value_counts().items()}
    m["elite_parent_f_r_min"] = float(pfeas["f_r"].min()) if len(pfeas) else None

    # ---- DB truth (dbx frontier table) -------------------------------------
    ft = cfg.p(cfg.frontier_table)
    m["sources"]["db"] = cfg.frontier_table
    if ft.exists():
        t = pd.read_csv(ft)
        sub = t[(t["pair"] == pair) & (t["feed"] == feed)]
        if len(sub):
            best = sub.loc[sub["F_r_min"].idxmin()]
            m["db_n_cores"] = int(sub["n_cores"].sum())
            m["db_f_r_min"] = float(best["F_r_min"])
            m["db_best_efpd"] = float(best["best_EFPD"])
            m["db_best_split"] = float(best["best_split"])
            m["db_best_n1"] = int(best["best_n_type1"])
            m["db_best_n2"] = int(best["best_n_type2"])
            m["db_best_cbc"] = float(best["best_CBC_max"])
            m["db_best_f_q"] = float(best["best_F_q"])
            m["db_pin_node"] = (None if pd.isna(best["best_pinmax_node_GWd"])
                                else float(best["best_pinmax_node_GWd"]))
            m["db_pin_known"] = bool(best["best_pinmax_known"])
            # The cliff verdict is "EVERY split bucket is fully over" -> min, not max
            # (dbx_frontier_note_20260816.md sec. 5 reads `frac_node_ge75 = 1.0`
            # across the cell, and its four campaign-only cells carry False meaning
            # UNKNOWN, which is why the NaNs are dropped rather than counted as 0).
            frac = sub["frac_node_ge75"].dropna()
            m["db_frac_node_ge75"] = float(frac.min()) if len(frac) else None
            if m["db_frac_node_ge75"] == 1.0:
                m["notes"].append(
                    f"DB PIN-BURNUP CLIFF: EVERY DB core in this cell exceeds 75 GWd/tU node "
                    f"pin burnup (this cell's frontier core sits at "
                    f"{m.get('db_pin_node')} GWd/tU).  dbx_frontier_note_20260816.md sec. 5 "
                    f"calls that family a licensing dead end unless the ceiling is relaxed. "
                    f"min_fr_max_cycle does NOT gate pin burnup, so a low-F_r core found "
                    f"here may be undeliverable — registered in advance, not discovered after."
                )
        else:
            m["notes"].append("no DB frontier row for this cell (DB is silent here).")
    else:
        m["notes"].append(f"frontier table absent: {ft}")

    # ---- model predicted floor + bias + verdict (the scoping mesh) ---------
    verd = cfg.p(cfg.mesh_dir) / "cell_verdicts.csv"
    sel = cfg.p(cfg.mesh_dir) / "pair_selection.csv"
    m["sources"]["mesh"] = f"{cfg.mesh_dir}/cell_verdicts.csv"
    segment, same_pair = None, False
    if sel.exists():
        ps = pd.read_csv(sel)
        hit = ps[ps["pair"] == pair]
        if len(hit):
            segment, same_pair = float(hit.iloc[0]["e_target"]), True
    if segment is None and m["e_core"] is not None:
        # The mesh grid is indexed by enrichment segment, and `pair_selection.csv`
        # names ONE representative pair per segment.  For a pair the mesh never
        # selected we can still read the segment row, but it describes a DIFFERENT
        # pair at the same enrichment — a weaker prior, and it is labelled as such.
        segment = round(m["e_core"], 1)
        m["notes"].append(
            f"mesh prior is the segment-{segment} row, whose representative pair is NOT "
            f"{pair}; read it as an enrichment-level prior, not a cell-level one."
        )
    m["segment"] = segment
    m["mesh_prior_is_same_pair"] = same_pair
    if verd.exists() and segment is not None:
        cv = pd.read_csv(verd)
        row = cv[(cv["segment"].round(2) == round(segment, 2)) & (cv["feed"] == feed)]
        if len(row):
            r = row.iloc[0]
            def _f(k):
                v = r.get(k)
                return None if v is None or pd.isna(v) else float(v)
            m["mesh_min_pred_f_r"] = _f("mesh_min_pred_f_r")     # THE model's predicted floor
            m["mesh_n_feasible"] = (None if pd.isna(r.get("mesh_n_feasible"))
                                    else int(r["mesh_n_feasible"]))
            m["f_r_bias_tail"] = _f("f_r_bias_tail")
            m["corrected_floor"] = _f("corrected_floor")
            m["gap_total"] = _f("gap_total")
            m["gap_data"] = _f("gap_data")
            m["gap_pool"] = _f("gap_pool")
            m["gap_search"] = _f("gap_search")
            m["mesh_verdict"] = str(r.get("verdict"))
        else:
            m["notes"].append(f"cell not in the mesh grid (segment={segment}, feed={feed}).")
    elif segment is None:
        m["notes"].append(f"pair {pair} not in pair_selection.csv — no mesh prior.")

    # ---- derived deck knobs ------------------------------------------------
    m.update(derive_deck_knobs(m))
    return m


def derive_deck_knobs(m: dict[str, Any]) -> dict[str, Any]:
    """The four cell-specific knobs, derived by RULE from the marks.

    Each rule is stated so a reader can check it against the hand-built f113 deck:

    ``cycle_target_efpd``   = the DB-truth core's own EFPD (report-only; gates
                              nothing on ``min_fr_max_cycle``).  f113: 659.7. Same.
    ``cycle_tolerance_efpd``= widened to span the cell's whole observed cyclen
                              range, rounded up to 5.  f113 rule gives 45 where the
                              human wrote 30; 30 did NOT actually span 617.5-651.
                              The rule is the more honest of the two and neither
                              gates anything.
    ``near_miss_f_r``       = midpoint of the store floor and the bias-corrected
                              mesh floor, clamped clear of both.  f113 rule gives
                              1.66 where the human chose 1.65 — same reasoning
                              ("below the store floor, above the corrected floor"),
                              0.01 apart, and it is a search knob that gates nothing.
    ``random_seed``         = deterministic per cell, never a spent seed.
    """

    store_floor = m.get("store_f_r_floor")
    db_floor = m.get("db_f_r_min")
    corrected = m.get("corrected_floor") or db_floor
    out: dict[str, Any] = {}

    # cycle readouts (record-only)
    target_efpd = m.get("db_best_efpd")
    if target_efpd is None:
        lo, hi = m.get("store_cyclen_min"), m.get("store_cyclen_max")
        target_efpd = round((lo + hi) / 2.0, 1) if lo and hi else 625.0
    out["cycle_target_efpd"] = float(target_efpd)
    lo, hi = m.get("store_cyclen_min"), m.get("store_cyclen_max")
    span = 30.0
    if lo is not None and hi is not None:
        span = max(30.0, target_efpd - lo, hi - target_efpd)
    out["cycle_tolerance_efpd"] = float(5.0 * ((span + 4.999) // 5))

    # near-miss parent admission bound.  Two properties, in priority order:
    #   (a) STRICTLY BELOW the store floor, so the arm cannot fill with the very
    #       produce rows the campaign is trying to beat;
    #   (b) not so low that it can never arm — it must sit at or above the
    #       bias-corrected reachable floor, less a small margin.
    # In a cell where we already match the DB (store ~= corrected) (a) dominates
    # and the bound lands just under our own floor, which is the correct reading:
    # "genuinely new territory" there IS below our own best.
    if store_floor is not None and corrected is not None:
        mid = (store_floor + corrected) / 2.0
        nm = min(mid, store_floor - 0.02)
        nm = max(nm, corrected - 0.02)
        out["near_miss_f_r"] = float(round(nm, 2))
    elif store_floor is not None:
        out["near_miss_f_r"] = float(round(store_floor - 0.02, 2))
    else:
        out["near_miss_f_r"] = 1.65
    out["random_seed"] = derive_seed(m["pair"], int(m["feed"]))
    return out


def read_champion(cfg: AutoengConfig) -> str:
    """The champion pointer.  ONE source of truth: ``data/curriculum/state.json``."""

    p = cfg.p(cfg.champion_state)
    state = json.loads(p.read_text(encoding="utf-8"))
    return str(state["champion_model_dir"])


def next_arm(champion: str) -> tuple[str, str, str]:
    """``data/models/s1g`` -> (arm ``s1h``, split ``S1h``, parent split ``S1g``).

    The S1x round names are a single-letter sequence; ``build_split_S1b.py`` is
    fully parameterised by ``--parent``/``--name`` so no new script is needed.
    """

    name = Path(champion).name
    if len(name) == 3 and name.startswith("s1") and name[2].isalpha():
        nxt = name[:2] + chr(ord(name[2]) + 1)
        return nxt, "S1" + nxt[2], "S1" + name[2]
    raise ValueError(
        f"cannot derive the next S1x arm from champion {champion!r}; name the arm "
        f"explicitly before running the RETRAIN stage."
    )


# --------------------------------------------------------------------------- #
# precheck — resolver ladder dry-run + support + fleet
# --------------------------------------------------------------------------- #
def resolve_assets(cfg: AutoengConfig, target: Target) -> dict[str, Any]:
    """Dry-run the CaseAssetResolver ladder for this cell (no MASTER, no writes)."""

    sys.path.insert(0, str(cfg.root))
    from lpopt.config import load_config
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.search.resolver import build_case_resolver
    from lpopt.vendor.masterrl.domain import CaseKey

    deck = load_config(cfg.p(cfg.parent_deck))
    # PRECHECK runs LOCALLY, so point the resolver at the local package copy — the
    # deck's own package_root is the kit-relative one.
    deck.verify.package_root = str(cfg.p(cfg.package_root))
    lib = FuelLibrary.from_parquet(cfg.p(cfg.store_dir) / "fuel_types.parquet")
    resolver = build_case_resolver(deck, lib, target.library)
    res = resolver.resolve(CaseKey(target.pair, int(target.feed)))

    def _rel(p) -> str | None:
        if p is None:
            return None
        try:
            return str(Path(p).resolve().relative_to(cfg.root.parent.parent))
        except ValueError:
            return str(p)

    return {
        "restart": (res.restart_path.name if res.restart_path else None),
        "restart_path": _rel(res.restart_path),
        "template": _rel(res.template_deck_path),
        "fallback_level": int(res.fallback_level),
        "restart_provenance": res.restart_provenance,
        "notes": list(res.notes),
        "resolvable": res.restart_path is not None and res.template_deck_path is not None,
    }


def precheck(cfg: AutoengConfig, target: Target) -> dict[str, Any]:
    """Stage 0.  Everything measurable before a MASTER call is spent."""

    out: dict[str, Any] = {"cell": target.cell_id, "library": target.library,
                           "blockers": [], "warnings": []}
    if target.is_new_assembly:
        out["kind"] = "new_assembly"
        spec = cfg.p(target.new_assembly)
        out["spec_path"] = str(spec)
        out["spec_present"] = spec.exists()
        out["blockers"].append(
            "NEW ASSEMBLY: the DeCART2D -> library -> bootstrap chain (paramA precedent) "
            "is expensive and stays human-gated.  autoeng v1 stops here."
        )
        return out

    out["kind"] = "existing_library"
    out["assets"] = resolve_assets(cfg, target)
    if not out["assets"]["resolvable"]:
        out["blockers"].append(f"assets unresolvable: {out['assets']['notes']}")
    if out["assets"]["fallback_level"] >= 3:
        out["blockers"].append(
            f"restart resolves at level {out['assets']['fallback_level']} (cross-pair). "
            f"Its burnt types are absent from this pair's %LPD_B&C and MASTER dies at "
            f"INITIALIZE — the campaign would burn its budget on errors."
        )

    marks = measure_marks(cfg, target)
    # The prereg renderer quotes the resolved assets, so they travel INSIDE marks
    # rather than through a module-level cache.
    marks["assets"] = out["assets"]
    out["marks"] = marks
    out["champion"] = read_champion(cfg)

    if marks["trust_region_support"] < 30:
        out["warnings"].append(
            f"trust-region support {marks['trust_region_support']} < the deck's n_min=30: "
            f"frontier sigma inflation will fire on nearly every candidate."
        )
    if marks["elite_parents_feasible"] < 8:
        out["warnings"].append(
            f"only {marks['elite_parents_feasible']} constraint-feasible {target.pair} rows "
            f"exist at any feed — the cross-feed elite seeding that carried f113 is thin here."
        )
    if marks.get("db_frac_node_ge75") == 1.0:
        out["warnings"].append(marks["notes"][-1])
    if marks["store_convergence_rate"] is not None and marks["store_convergence_rate"] < 0.5:
        out["warnings"].append(
            f"historical convergence in this cell is {marks['store_convergence_rate']:.1%}; "
            f"expect ~{int(target.budget * marks['store_convergence_rate'])} usable labels "
            f"from {target.budget} calls, not {target.budget}."
        )
    return out


# --------------------------------------------------------------------------- #
# deck + prereg generation
# --------------------------------------------------------------------------- #
def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        if any(isinstance(x, dict) for x in v):
            raise TypeError("array-of-tables is not supported by the autoeng deck emitter")
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot emit {type(v).__name__} to TOML")


def _toml_dumps(data: dict[str, Any], prefix: str = "") -> str:
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines: list[str] = []
    for k, v in scalars.items():
        lines.append(f"{k} = {_toml_value(v)}")
    for k, v in tables.items():
        name = f"{prefix}{k}"
        lines.append("")
        lines.append(f"[{name}]")
        body = _toml_dumps(v, prefix=f"{name}.")
        if body:
            lines.append(body)
    return "\n".join(lines).strip("\n")


#: Keys the automation is allowed to change per cell.  Anything else is carried
#: from the parent deck byte-for-byte — that is what makes this "the f113 recipe".
CELL_OVERRIDE_KEYS = (
    ("flow", "title"), ("flow", "random_seed"),
    ("case", "pair"), ("case", "feed"),
    ("model", "model_dir"), ("model", "library_id"),
    ("acquisition", "budget"),
    ("acquisition", "cycle_target_efpd"), ("acquisition", "cycle_tolerance_efpd"),
    ("search", "near_miss_f_r"),
)


def build_deck(cfg: AutoengConfig, target: Target, marks: dict[str, Any],
               champion: str) -> tuple[str, dict[str, Any]]:
    """Render the cell's campaign deck: generated prereg header + the f113 knobs.

    Returns ``(text, overrides)``.  Every knob NOT in ``overrides`` is the parent
    deck's value, round-tripped through tomllib, so the proven recipe is carried
    rather than retyped.
    """

    parent_path = cfg.p(cfg.parent_deck)
    data = tomllib.loads(parent_path.read_text(encoding="utf-8"))
    for sec in DROP_SECTIONS:
        data.pop(sec, None)

    title = (f"min_fr_max_cycle AUTO-OPEN — {target.pair}/f{target.feed} "
             f"({target.library}, e_core {marks.get('e_core')}), champion "
             f"{Path(champion).name}, PURE F_r min, NO cycle band")
    overrides: dict[tuple[str, str], Any] = {
        ("flow", "title"): title,
        ("flow", "random_seed"): int(marks["random_seed"]),
        ("case", "pair"): target.pair,
        ("case", "feed"): int(target.feed),
        ("model", "model_dir"): champion,
        ("model", "library_id"): target.library,
        ("acquisition", "budget"): int(target.budget),
        ("acquisition", "cycle_target_efpd"): float(marks["cycle_target_efpd"]),
        ("acquisition", "cycle_tolerance_efpd"): float(marks["cycle_tolerance_efpd"]),
        ("search", "near_miss_f_r"): float(marks["near_miss_f_r"]),
    }
    for (sec, key), val in overrides.items():
        if sec not in data:
            raise ValueError(f"parent deck {parent_path} has no [{sec}] section")
        data[sec][key] = val

    header = render_prereg(cfg, target, marks, champion, parent_path, overrides)
    body = _toml_dumps(data)
    return header + "\n" + body + "\n", {f"{s}.{k}": v for (s, k), v in overrides.items()}


def render_prereg(cfg: AutoengConfig, target: Target, marks: dict[str, Any],
                  champion: str, parent_path: Path, overrides: dict) -> str:
    """The pre-registration, as the deck's own comment header (the f113 convention).

    Written BEFORE the campaign, from measured marks only.  This is the honesty
    discipline the automation must not lose: a mark that is not in this block was
    not pinned in advance and cannot be claimed afterwards.
    """

    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    n = lambda k, f="{}": ("n/a" if marks.get(k) is None else f.format(marks[k]))  # noqa: E731
    L: list[str] = []
    A = L.append
    A("#" * 79)
    A(f"# AUTO-GENERATED by autoeng.py — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"# CELL {target.pair} / feed {target.feed} / {target.library}   champion {champion}")
    A(f"# parent recipe: {parent_path.name}  sha256 {parent_sha}")
    A("#" * 79)
    A("# PRE-REGISTRATION — written BEFORE launch, from measured marks only.")
    A("#" * 79)
    A("#")
    A("# OBJECTIVE (registered, one line):")
    A("#   PURE F_r MINIMISATION UNDER SAFETY GATES ONLY.  No cycle-length target and")
    A("#   no cycle-length band; cyclen is RECORDED on every row and is the objective's")
    A("#   subordinate tie-break, nothing more.  (User directive 2026-08-16.)")
    A("#")
    A("# ------------------------------------------------------------- THE MARKS --")
    A("# Measured, not assumed.  Every source is named so each number re-derives.")
    A("#")
    A(f"# 1. OUR STORE'S FLOOR IN THIS CELL  ({marks['sources']['store']})")
    A(f"#      rows {marks['store_rows']}   converged {marks['store_converged']}   "
      f"constraint-feasible {marks['store_feasible']}")
    A(f"#      F_r floor {n('store_f_r_floor', '{:.4f}')}   p10 {n('store_f_r_p10', '{:.4f}')}")
    A(f"#      cyclen {n('store_cyclen_min', '{:.1f}')} - {n('store_cyclen_max', '{:.1f}')} EFPD")
    A(f"#      historical convergence {n('store_convergence_rate', '{:.1%}')} -> expect "
      f"~{int(target.budget * (marks.get('store_convergence_rate') or 1.0))} usable labels "
      f"from {target.budget} calls")
    A(f"#      trust-region support (feed+-4, e_core+-0.10): {marks['trust_region_support']}")
    A("#")
    A(f"# 2. DB TRUTH  ({marks['sources'].get('db', 'n/a')})")
    if marks.get("db_f_r_min") is not None:
        A(f"#      {marks['db_n_cores']} cores   F_r min {marks['db_f_r_min']:.4f}   "
          f"EFPD {marks['db_best_efpd']:.1f}   CBC {marks['db_best_cbc']:.0f}   "
          f"F_q {marks['db_best_f_q']:.3f}")
        A(f"#      best composition split {marks['db_best_split']:.3f} "
          f"({marks['db_best_n1']} x {marks['db_best_n2']})   "
          f"node pin burnup {n('db_pin_node', '{:.1f}')} GWd/tU "
          f"(known={marks.get('db_pin_known')})")
        if marks.get("store_f_r_floor") is not None:
            A(f"#      THE DATA GAP IS {marks['store_f_r_floor'] - marks['db_f_r_min']:+.4f} F_r "
              f"(store floor - DB truth).")
    else:
        A("#      the database is SILENT at this cell — no DB mark to beat.")
    A("#")
    A(f"# 3. THE MODEL'S OWN PREDICTED FLOOR  ({marks['sources'].get('mesh', 'n/a')})")
    if marks.get("mesh_min_pred_f_r") is not None:
        A(f"#      mesh_min_pred_f_r {marks['mesh_min_pred_f_r']:.4f}   "
          f"mesh_n_feasible {marks.get('mesh_n_feasible')}")
        A(f"#      f_r_bias_tail {n('f_r_bias_tail', '{:+.4f}')}   "
          f"corrected_floor {n('corrected_floor', '{:.4f}')}")
        A(f"#      gap_total {n('gap_total', '{:+.4f}')}   gap_data {n('gap_data', '{:+.4f}')}   "
          f"gap_pool {n('gap_pool', '{:+.4f}')}   gap_search {n('gap_search', '{:+.4f}')}")
        A(f"#      verdict: {marks.get('mesh_verdict')}")
    else:
        A("#      this cell is not on the mesh grid — no model prior is registered.")
    A("#")
    A("# 4. THE ELITE PARENT SET (cross-feed transfer, the f113 mechanism)")
    A(f"#      {marks['elite_parents_feasible']} constraint-feasible {target.pair} rows exist "
      f"across all feeds; by feed {marks['elite_parent_feeds']}")
    A(f"#      best parent F_r {n('elite_parent_f_r_min', '{:.4f}')}.  `_case_store_rows` "
      f"filters by case_pair ONLY (no feed filter), so `_store_elites` sees them all and")
    A("#      `_parent_to_genome` feed-morphs each one to this cell's n_fresh.")
    A("#")
    A("# ------------------------------------------------------------- SUCCESS ----")
    A("#   PRIMARY   a MASTER-verified constraint-feasible core (F_r <= 1.55, CBC <= 1600,")
    A("#             F_q <= 2.41, |AO| <= 0.30) at this cell.")
    if marks.get("db_f_r_min") is not None:
        A(f"#   SECONDARY F_r <= {marks['db_f_r_min']:.4f} — matching or beating the DB's own best")
        A("#             core here with a loading pattern we actually hold.")
    if marks.get("store_f_r_floor") is not None:
        A(f"#   PARTIAL   no feasible row, but the cell's floor moves below "
          f"{marks['store_f_r_floor']:.4f}.  Every 0.01")
        A("#             of that is a FRONTIER LABEL in a region the champion has never seen.")
        A(f"#   NULL      the floor does not move below ~{marks['store_f_r_floor']:.2f} in "
          f"{target.budget} calls.  PUBLISHABLE.")
    A("#             The reading is NOT that the cell is closed; it is that cross-feed")
    A("#             elite transfer is insufficient here and the cell needs its own")
    A("#             label -> retrain -> re-search loop.")
    if marks.get("notes"):
        A("#")
        A("# --------------------------------------------------- REGISTERED CAVEATS --")
        A("#   Known BEFORE launch.  None of these is allowed to appear for the first")
        A("#   time in the results write-up.")
        for note in marks["notes"]:
            for i, chunk in enumerate(_wrap(note, 72)):
                A(f"#   {'* ' if i == 0 else '  '}{chunk}")
    A("#")
    A("# ------------------------------------------------------- ASSET RESOLUTION --")
    a = marks.get("assets") or {}
    if a:
        A(f"#   restart  {a.get('restart')}  (level {a.get('fallback_level')}, "
          f"provenance {a.get('restart_provenance')})")
        A(f"#   template {a.get('template')}")
        A("#   PREDICTED IN ADVANCE: every converged row must carry this provenance.")
    A("#")
    A("# --------------------------------------------------- DERIVED DECK KNOBS ---")
    A("#   Rules are in autoeng.derive_deck_knobs; only these differ from the parent.")
    for k, v in overrides.items():
        A(f"#   {k[0]}.{k[1]} = {v!r}")
    A(f"#   [constraints] DROPPED: inert on min_fr_max_cycle "
      f"(fpcamp_N1N2_f113_results_20260816.md, defect 2).")
    A("#")
    A("# CONSTRAINTS HONOURED: 198 / 181 untouched.  The canonical local store is")
    A("# READ-ONLY until the post-campaign merge.")
    A("#" * 79)
    A("")
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out or [""]


def render_probe_script(cfg: AutoengConfig, target: Target, marks: dict[str, Any],
                        champion: str, deck_name: str) -> str:
    """Stage 1: the blind-transfer probe, composed from ``_phase_blind_probe``.

    It measures the champion's a-priori skill in a cell it has never trained on:
    the model predicts FIRST, MASTER then labels the SAME patterns, and the
    per-target (mae, bias, rmse, spearman, mean_abs_z, cov1, cov2) table is the
    record.  Written as a standalone script so a human can read the 8 calls it is
    about to spend before they are spent.
    """

    e = marks.get("e_core") or 5.4
    lo = round(0.25 * int(e / 0.25), 2)
    hi = round(lo + 0.25, 2)
    cid = f"{target.pair}_f{target.feed}"
    return f'''"""autoeng blind-transfer probe — {cid}.  {target.probe_budget} MASTER calls.

Composed from lpopt.curriculum.CurriculumDriver._phase_blind_probe (plan sec 12.3).
It uses its OWN state_dir, so the real data/curriculum/state.json is never touched.
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(".").resolve()))
from lpopt.config import load_config
from lpopt.curriculum import CurriculumDriver

STATE = pathlib.Path(r"autoeng_probe/{cid}")
cfg = load_config(pathlib.Path(r"{deck_name}"))
drv = CurriculumDriver(cfg, progress=False, state_dir=STATE)
if drv.state_path.exists():
    drv._load_state()
else:
    drv._init_state()
drv.curr.probe_size = {target.probe_budget}
drv.state["champion_model_dir"] = r"{champion}"
cid = "{cid}"
drv.state["cells"][cid] = {{
    "band": [{lo}, {hi}], "feed": {int(target.feed)}, "pairs": ["{target.pair}"],
    "library_id": "{target.library}", "ring": 0, "phase": "blind_probe",
    "budget": {{"master_calls": 0, "wall_s": 0.0}},
}}
drv._phase_blind_probe(cid)
out = drv.cell_dir(cid) / "blind_probe.json"
print("PROBE OK ->", out)
print(json.dumps(json.loads(out.read_text(encoding="utf-8"))["per_target"], indent=1))
'''


def render_run_bat(cfg: AutoengConfig, target: Target, deck_name: str, tag: str,
                   argv_tail: str) -> str:
    kit = cfg.fleet.campaign_kit.replace("/", "\\")
    return f"""@echo off
REM autoeng — {tag} — {target.pair}/f{target.feed}/{target.library} on the campaign box.
REM Generated by autoeng.py; do not hand-edit (the launcher gates on the deck hash).
setlocal
chcp 65001 > nul
set LPOPT_WORKER=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "{kit}"
set PY={cfg.fleet.campaign_python.replace('/', chr(92))}
"%PY%" -u {argv_tail} > {tag}_out.log 2>&1
echo %ERRORLEVEL% > {tag}_rc.txt
endlocal
"""


def render_launch_ps1(cfg: AutoengConfig, target: Target, deck_name: str, tag: str,
                      deck_sha: str, champion: str, *, fresh_run_dir: str = "") -> str:
    """The arming script.  Carries the three gates the f113 launcher proved out:
    a BUSY gate (refuse rather than stack a second MASTER queue), a DECK-HASH gate
    (a wrong or truncated deck is cheaper to catch here than 100 calls later), and
    a PRECONDITION gate (champion + store present).  Launch is
    ``Invoke-CimMethod Win32_Process Create`` with a literal cmd.exe path —
    ``schtasks`` no-ops silently on this fleet.
    """

    kit = cfg.fleet.campaign_kit.replace("/", "\\")
    fresh = ""
    if fresh_run_dir:
        fresh = (f"Remove-Item (Join-Path $k '{fresh_run_dir}') -Recurse -Force "
                 f"-EA SilentlyContinue\n")
    return f"""# autoeng {tag} — runs ON the campaign box.  Generated; do not hand-edit.
$ErrorActionPreference = 'Continue'
$k = '{kit}'

# -- busy gate: REFUSE rather than stack a second MASTER queue ---------------
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
          Where-Object {{ $_.CommandLine -match 'lpopt|ablation|batchswap|autoeng' }}).Count
$mc = @(Get-Process master4.0m4_r1 -EA SilentlyContinue).Count
if ($busy -gt 0 -or $mc -gt 0) {{
  Write-Output ("{tag} REFUSED: box busy (lpopt_python=$busy master=$mc)"); exit 1
}}

# -- deck hash gate: the deck must be the one the pre-registration hashed -----
$deck = Join-Path $k '{deck_name}'
if (-not (Test-Path $deck)) {{ Write-Output "{tag} REFUSED: deck not found"; exit 1 }}
$want = '{deck_sha.upper()}'
$got  = (Get-FileHash -Algorithm SHA256 $deck).Hash
if ($got -ne $want) {{
  Write-Output "{tag} REFUSED: deck sha256 mismatch"
  Write-Output "  expected $want"
  Write-Output "  found    $got"
  exit 1
}}

# -- precondition gate -------------------------------------------------------
if (-not (Test-Path (Join-Path $k '{champion}\\ensemble.json'))) {{
  Write-Output "{tag} REFUSED: champion {champion} not on the kit"; exit 1 }}
if (-not (Test-Path (Join-Path $k 'data\\store\\records.parquet'))) {{
  Write-Output "{tag} REFUSED: store missing"; exit 1 }}
$bat = Join-Path $k 'run_{tag}.bat'
if (-not (Test-Path $bat)) {{ Write-Output "{tag} REFUSED: run bat not at the kit root"; exit 1 }}

# -- fresh run dir: a stale partial would be resumed and silently shrink the run
{fresh}Remove-Item (Join-Path $k '{tag}_rc.txt')  -Force -EA SilentlyContinue
Remove-Item (Join-Path $k '{tag}_out.log') -Force -EA SilentlyContinue

$cmdline = '"C:\\Windows\\System32\\cmd.exe" /c "' + $bat + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{
        CommandLine = $cmdline; CurrentDirectory = $k }}
Write-Output ("{tag} Win32_Process Create: ReturnValue=" + $r.ReturnValue +
              " ProcessId=" + $r.ProcessId)
if ($r.ReturnValue -ne 0) {{ Write-Output "{tag} LAUNCH FAILED"; exit 1 }}
Write-Output "{tag} armed. rc=$k\\{tag}_rc.txt log=$k\\{tag}_out.log"
"""


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    name: str
    stage: str
    where: str                    # "inproc" | "local" | "199" | "238"
    what: str                     # Korean, for the lab notebook
    argv: tuple[str, ...] = ()
    master_calls: int = 0
    writes: tuple[str, ...] = ()
    gate: str = ""                # gate name evaluated AFTER this step
    poll_argv: tuple[str, ...] = ()
    poll_until: str = ""
    poll_timeout_s: int = 0
    skip_if: str = ""             # a key in the run context; truthy -> skip


def plan_cell(cfg: AutoengConfig, target: Target, *, champion: str,
              arm: str = "", split: str = "", parent_split: str = "") -> list[Step]:
    """The concrete per-cell step list.  Every argv here is what would be executed."""

    cid = target.cell_id
    # Every command runs with cwd == repo root, so quote cell paths RELATIVE to it:
    # the absolute form contains spaces and Korean, and would need quoting in the
    # ssh/scp legs where the shell on the far side is not ours.
    cdir = Path(f"data/autoeng/{cfg.run_id}/cells/{cid}")
    deck_name = f"autoeng_{cid}.inp"
    tag = f"autoeng_{cid}"
    host = cfg.fleet.campaign_host
    kit = cfg.fleet.campaign_kit
    py = "python"
    arm_stamp = time.strftime("%Y%m%d")

    if target.is_new_assembly:
        return [
            Step("precheck", "0-precheck", "inproc",
                 f"신규 집합체 {cid}: 설계 체인 사전점검 (DeCART 미실행)",
                 writes=(str(cdir / "precheck.json"),), gate="new_assembly"),
        ]

    S: list[Step] = []
    A = S.append

    # ---- 0 PRECHECK --------------------------------------------------------
    A(Step("precheck", "0-precheck", "inproc",
           f"{cid}: 자산 해석 사다리 dry-run · 스토어 지지량 · DB/메시 사전분포 · 모델 예측 하한",
           writes=(str(cdir / "precheck.json"),)))
    A(Step("prereg", "0-precheck", "inproc",
           f"{cid}: 사전등록(prereg) + 캠페인 덱 생성 및 sha256 고정 — MASTER 호출 이전",
           writes=(str(cdir / "prereg.md"), str(cdir / deck_name))))
    A(Step("arm_scripts", "0-precheck", "inproc",
           f"{cid}: probe 스크립트 · run_*.bat · launch_*.ps1 생성 (busy/해시/사전조건 게이트 포함)",
           writes=(str(cdir / f"probe_{cid}.py"), str(cdir / f"run_{tag}.bat"),
                   str(cdir / f"launch_{tag}.ps1"))))
    A(Step("ship_kit", "0-precheck", "local",
           f"{cid}: 덱·스크립트를 캠페인 박스 kit로 전송",
           argv=("scp", str(cdir / deck_name), str(cdir / f"run_{tag}.bat"),
                 str(cdir / f"launch_{tag}.ps1"), str(cdir / f"probe_{cid}.py"),
                 str(cdir / f"run_{tag}_probe.bat"), str(cdir / f"launch_{tag}_probe.ps1"),
                 f"{host}:{kit}/")))

    # ---- 1 PROBE (blind transfer) -----------------------------------------
    A(Step("probe_launch", "1-probe", "199",
           f"{cid}: 블라인드 전이 프로브 {target.probe_budget}콜 — 모델이 먼저 예측, MASTER가 검증",
           argv=("ssh", host, f"powershell -NoProfile -ExecutionPolicy Bypass -File "
                              f"{kit}/launch_{tag}_probe.ps1"),
           master_calls=target.probe_budget))
    A(Step("probe_wait", "1-probe", "199", f"{cid}: 프로브 완료 대기 (rc 파일 폴링)",
           poll_argv=("ssh", host, f"powershell -NoProfile -Command \"Get-Content "
                                   f"{kit}/{tag}_probe_rc.txt -EA SilentlyContinue\""),
           poll_until="0", poll_timeout_s=7200))
    A(Step("probe_pull", "1-probe", "local", f"{cid}: blind_probe.json 회수",
           argv=("scp", f"{host}:{kit}/autoeng_probe/{cid}/cells/{cid}/blind_probe.json",
                 str(cdir / "blind_probe.json"))))
    A(Step("probe_readout", "1-probe", "inproc",
           f"{cid}: 챔피언의 사전(a-priori) 실력 판독 — target별 mae/bias/spearman/cov",
           writes=(str(cdir / "probe_readout.json"),)))

    # ---- 2 OPEN ------------------------------------------------------------
    A(Step("open_launch", "2-open", "199",
           f"{cid}: 개방 캠페인 {target.budget}콜 (f113 레시피 그대로: lambda=1000, 밴드 없음, "
           f"교차-feed 엘리트 모프, wave별 finetune)",
           argv=("ssh", host, f"powershell -NoProfile -ExecutionPolicy Bypass -File "
                              f"{kit}/launch_{tag}.ps1"),
           master_calls=target.budget))
    A(Step("open_wait", "2-open", "199", f"{cid}: 캠페인 완료 대기 (rc 파일 폴링)",
           poll_argv=("ssh", host, f"powershell -NoProfile -Command \"Get-Content "
                                   f"{kit}/{tag}_rc.txt -EA SilentlyContinue\""),
           poll_until="0", poll_timeout_s=172800))

    # ---- 3 HARVEST + MERGE -------------------------------------------------
    # The kit's run dir was 10.3 GB / 1340 files for f113 — pulling it whole is a
    # mistake.  Take the four artefacts the readout actually needs, then the kit's
    # data/ folder, which is what `merge-store --from` consumes.
    A(Step("harvest_run", "3-harvest", "local",
           f"{cid}: 캠페인 판독물 회수 (labels/state/status/report — 런디렉터리 전체가 아니라)",
           argv=("scp", f"{kit}/runs/{tag}/labels.jsonl", f"{kit}/runs/{tag}/state.json",
                 f"{kit}/runs/{tag}/status.json", f"{kit}/runs/{tag}/report.md",
                 str(cdir) + "/"),
           writes=(str(cdir / "state.json"),)))
    A(Step("harvest_data", "3-harvest", "local", f"{cid}: kit 스토어(store+maps) 회수",
           argv=("scp", "-r", f"{kit}/data", str(cdir) + "/"),
           writes=(str(cdir / "data"),)))
    A(Step("store_backup", "3-harvest", "inproc",
           f"{cid}: 정본 스토어 백업 (records.parquet / maps.npz .bak_pre_{cid})",
           writes=(f"{cfg.store_dir}/records.parquet.bak_pre_{cid}",)))
    A(Step("merge_dryrun", "3-harvest", "local", f"{cid}: merge-store dry-run 검수",
           argv=(py, "-m", "lpopt", "merge-store", "--input", cfg.main_deck,
                 "--from", str(cdir / "data"), "--dry-run")))
    A(Step("merge", "3-harvest", "local", f"{cid}: 정본 스토어 병합 (new/upgraded/duplicate 집계)",
           argv=(py, "-m", "lpopt", "merge-store", "--input", cfg.main_deck,
                 "--from", str(cdir / "data"))))

    # ---- 4 RETRAIN + GATE --------------------------------------------------
    arm = arm or "<arm>"
    split = split or "<split>"
    parent_split = parent_split or "<parent_split>"
    A(Step("retrain_prereg", "4-retrain", "inproc",
           f"{cid}: 재학습 라운드 사전등록 ({arm}) — 입력 sha256 고정 (records/maps/fuel_types/부모 split)",
           writes=(f"data/reports/ab2_addendum_{arm.upper()}_{arm_stamp}.md",)))
    A(Step("split_dryrun", "4-retrain", "local",
           f"{cid}: 증분 split {parent_split} -> {split} 검증 (기본이 dry-run)",
           argv=(py, "build_split_S1b.py", "--parent", parent_split, "--name", split,
                 "--holdout-new-campaigns")))
    A(Step("split_write", "4-retrain", "local", f"{cid}: split 기록 (모든 MUST_HOLD 통과 시에만)",
           argv=(py, "build_split_S1b.py", "--parent", parent_split, "--name", split,
                 "--holdout-new-campaigns", "--write"),
           writes=(f"data/splits/{split}.json",)))
    A(Step("train_push", "4-retrain", "238", f"{cid}: 238로 소스+스토어+split 푸시",
           argv=(py, "-m", "lpopt.remote", "--input", cfg.fleet.train_deck, "push")))
    A(Step("train_launch", "4-retrain", "238",
           f"{cid}: GPU1 학습 시작 ({arm}) — s1f/s1g 레시피에서 --split만 교체",
           argv=(py, "-m", "lpopt.remote", "--input", cfg.fleet.train_deck, "train",
                 "--ts", arm, "--", *[a.format(split=split) for a in TRAIN_RECIPE])))
    A(Step("train_wait", "4-retrain", "238", f"{cid}: 학습 완료 대기 (DONE/rc 폴링)",
           poll_argv=(py, "-m", "lpopt.remote", "--input", cfg.fleet.train_deck,
                      "status", "--ts", arm),
           poll_until="DONE", poll_timeout_s=172800))
    A(Step("train_pull", "4-retrain", "local", f"{cid}: 학습 산출물 회수 -> data/models/{arm}",
           argv=(py, "-m", "lpopt.remote", "--input", cfg.main_deck, "pull", "--ts", arm),
           writes=(f"data/models/{arm}",)))
    A(Step("gate_check", "4-retrain", "local",
           f"{cid}: 무회귀 + legacy-tail 게이트 (판정만, 승격 없음)",
           argv=(py, "-m", "lpopt", "gate-promote", "--input", cfg.main_deck,
                 "--prev", champion, "--new", f"data/models/{arm}",
                 "--out", f"data/reports/gate_{arm}_checkonly.json", "--check-only"),
           writes=(f"data/reports/gate_{arm}_checkonly.json",),
           gate="retrain_promote_fail"))
    A(Step("gate_promote", "4-retrain", "local",
           f"{cid}: PASS일 때만 챔피언 승격 (state.json + lpopt.inp 원자적 갱신)",
           argv=(py, "-m", "lpopt", "gate-promote", "--input", cfg.main_deck,
                 "--prev", champion, "--new", f"data/models/{arm}",
                 "--out", f"data/reports/gate_{arm}.json"),
           writes=(f"data/reports/gate_{arm}.json",),
           skip_if="gate_failed"))

    # ---- 5 MAP UPDATE ------------------------------------------------------
    # Skipped when the gate did not promote: the pool-only effect on the mesh was
    # measured at -0.0036 mean (comparison_readout.md sec. 10.2), i.e. inside noise.
    old = Path(champion).name
    A(Step("mesh_baseline", "5-map", "inproc",
           f"{cid}: 메시 기준선 보존 (model_bias/cell_verdicts -> *_{old}.csv) — "
           f"mesh_vs_db.py가 무접미사 파일을 덮어쓰기 때문",
           skip_if="gate_failed",
           writes=(f"{cfg.mesh_dir}/cell_verdicts_{old}.csv",)))
    A(Step("mesh_recompute", "5-map", "199",
           f"{cid}: 메시 재계산 (모델 전용, MASTER 0콜) — 챔피언이 바뀌었으므로 필수",
           argv=("ssh", host, f"cd {kit} && python scoping_mesh.py --model {arm}"),
           skip_if="gate_failed"))
    A(Step("mesh_pull", "5-map", "local", f"{cid}: mesh_nodes/mesh_pareto 회수",
           argv=("scp", f"{host}:{kit}/{cfg.mesh_dir}/mesh_nodes.csv",
                 f"{host}:{kit}/{cfg.mesh_dir}/mesh_pareto.csv",
                 str(cfg.p(cfg.mesh_dir))),
           skip_if="gate_failed"))
    A(Step("mesh_verdict", "5-map", "local", f"{cid}: DB 대비 판정표 갱신 (cell_verdicts.csv)",
           argv=(py, "mesh_vs_db.py", "--model", arm),
           writes=(f"{cfg.mesh_dir}/cell_verdicts.csv",), skip_if="gate_failed"))
    A(Step("mesh_figure", "5-map", "local", f"{cid}: 지도 렌더",
           argv=(py, "scoping_mesh.py", "--figure-only", "--model", arm),
           writes=(f"{cfg.mesh_dir}/scoping_mesh.png",), skip_if="gate_failed"))

    # ---- 6 REPORT ----------------------------------------------------------
    A(Step("cell_report", "6-report", "inproc",
           f"{cid}: 셀 보고서 + 실험노트(AUTOENG_LOG.md) 기록 — 등록한 모든 마크 대비 판정",
           writes=(str(cdir / "report.md"), cfg.notebook)))
    return S


# --------------------------------------------------------------------------- #
# transfer-aware cell ordering
# --------------------------------------------------------------------------- #
#: A cell counts as OPENED once it holds this many MASTER-verified constraint-
#: feasible rows.  1 is noise (several cells hold a single lucky produce row);
#: the f113 campaign delivered 41, and the cells that seeded it held 12-14.
OPENED_MIN_FEASIBLE = 5


def opened_cells_from_store(cfg: AutoengConfig, library: str = "ga80") -> list[str]:
    """Cells the programme has actually opened, read off the canonical store.

    autoeng's own log only knows about cells IT opened; the transfer argument is
    about the labels that exist, whoever produced them.  N1_N2/f113 was opened by
    hand on 2026-08-16 and must count.
    """

    import pandas as pd

    recs = cfg.p(cfg.store_dir) / "records.parquet"
    if not recs.exists():
        return []
    df = pd.read_parquet(recs, columns=["library_id", "case_pair", "feed", "converged",
                                        "valid", "f_r", "cbc_max", "f_q", "ao_abs"])
    g = df[(df["library_id"] == library)
           & df["converged"].fillna(False).astype(bool)
           & df["valid"].fillna(False).astype(bool)]
    f = g[(g["f_r"] <= 1.55) & (g["cbc_max"] <= 1600.0)
          & (g["f_q"] <= 2.41) & (g["ao_abs"].abs() <= 0.30)]
    counts = f.groupby(["case_pair", "feed"]).size()
    return [f"{p}_f{int(fd)}" for (p, fd), n in counts.items() if n >= OPENED_MIN_FEASIBLE]


def order_targets(cfg: AutoengConfig, targets: Sequence[Target],
                  opened: Sequence[str] = ()) -> list[Target]:
    """Nearest-neighbour expansion from already-opened cells, DB-frontier tie-break.

    Why nearest-neighbour: ``comparison_readout.md`` sec. 10.3 measured that the
    f113 frontier labels moved the NEIGHBOUR feeds' pessimism by -0.041 (f105) and
    -0.044 (f109), i.e. **82-86 % model-attributed**, and f109's drop was larger
    than the labelled column's own.  So an adjacent cell is genuinely cheaper, and
    the ordering is a measured lever, not an aesthetic.

    Tie-break: ``dbx_frontier_note_20260816.md`` sec. 4 — lower ``F_r_min`` first,
    with a hard demotion for the pin-burnup cliff (sec. 5, ``frac_node_ge75 == 1``).
    """

    import pandas as pd

    ft = cfg.p(cfg.frontier_table)
    tbl = pd.read_csv(ft) if ft.exists() else None
    opened_keys = []
    for cid in opened:
        if "_f" in cid:
            p, f = cid.rsplit("_f", 1)
            if f.isdigit():
                opened_keys.append((p, int(f)))

    def distance(t: Target) -> int:
        if t.is_new_assembly or not opened_keys:
            return 99
        best = 99
        for p, f in opened_keys:
            d = abs(int(t.feed) - f) // 4 if p == t.pair else 10 + abs(int(t.feed) - f) // 4
            best = min(best, d)
        return best

    def prior(t: Target) -> tuple[int, float]:
        if tbl is None or t.is_new_assembly:
            return (0, 9.0)
        sub = tbl[(tbl["pair"] == t.pair) & (tbl["feed"] == int(t.feed))]
        if not len(sub):
            return (0, 9.0)
        frac = sub["frac_node_ge75"].dropna()
        cliff = 1 if (len(frac) and float(frac.min()) == 1.0) else 0
        return (cliff, float(sub["F_r_min"].min()))

    def key(t: Target):
        c, fr = prior(t)
        # new-assembly targets last: they halt, so they must not block cheap cells.
        return (1 if t.is_new_assembly else 0, distance(t), c, fr, t.pair, int(t.feed))

    return sorted(targets, key=key)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    rc: int
    stdout: str = ""
    stderr: str = ""


def subprocess_runner(argv: Sequence[str], *, cwd: Path, timeout: int = 3600) -> RunResult:
    p = subprocess.run(list(argv), cwd=str(cwd), capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    return RunResult(p.returncode, p.stdout or "", p.stderr or "")


class AutoEngineer:
    """The loop.  One instance per invocation; all durable state is the JSONL log."""

    def __init__(self, cfg: AutoengConfig, *, dry_run: bool = False,
                 runner: Callable[..., RunResult] | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.runner = runner or subprocess_runner
        self._log = log or (lambda s: print(s, flush=True))
        self.state = StateLog(cfg.state_path)
        self.ctx: dict[str, Any] = {}          # per-cell run context (skip_if keys)

    def _opened(self) -> list[str]:
        """Opened cells = the store's evidence PLUS anything this run opened."""
        try:
            store = opened_cells_from_store(self.cfg)
        except Exception:                                            # noqa: BLE001
            store = []
        return sorted(set(store) | set(self.state.opened_cells()))

    # -- gates ------------------------------------------------------------- #
    def _pause(self, gate: str, cell: str, detail: str) -> None:
        self.state.append("gate_pause", cell=cell, step=gate, detail=detail)
        self._log(f"\n[autoeng] ⏸ HUMAN GATE '{gate}' at {cell}\n  {detail}\n"
                  f"  승인 후 재개:  python autoeng.py --config <cfg> --resume\n")

    def _check_budget(self, planned: int, cell: str) -> bool:
        spent = self.state.master_calls()
        if spent + planned > self.cfg.master_budget_total:
            if "budget_exceeded" in self.cfg.pause_for_approval:
                self._pause("budget_exceeded", cell,
                            f"spent {spent} + planned {planned} > cap "
                            f"{self.cfg.master_budget_total} MASTER calls")
                return False
            self.state.append("note", cell=cell,
                              detail=f"budget cap exceeded ({spent}+{planned}) — not gated")
        return True

    # -- inproc steps ------------------------------------------------------- #
    def _do_precheck(self, target: Target) -> dict[str, Any]:
        pc = precheck(self.cfg, target)
        cdir = self.cfg.cell_dir(target.cell_id)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "precheck.json").write_text(
            json.dumps(pc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        self.ctx["precheck"] = pc
        return pc

    def _do_prereg(self, target: Target) -> dict[str, Any]:
        pc = self.ctx["precheck"]
        marks, champion = pc["marks"], pc["champion"]
        text, overrides = build_deck(self.cfg, target, marks, champion)
        cdir = self.cfg.cell_dir(target.cell_id)
        deck_path = cdir / f"autoeng_{target.cell_id}.inp"
        deck_path.write_text(text, encoding="utf-8")

        # A generated deck that does not LOAD is worse than no deck: verify now.
        sys.path.insert(0, str(self.cfg.root))
        from lpopt.config import load_config
        loaded = load_config(deck_path)
        assert loaded.case.pair == target.pair and int(loaded.case.feed) == int(target.feed)
        assert loaded.acquisition.objective == "min_fr_max_cycle"
        assert loaded.acquisition.budget == target.budget

        sha = hashlib.sha256(deck_path.read_bytes()).hexdigest()
        # The pre-registration IS the deck header (the f113 convention).  prereg.md
        # is that header, verbatim, plus the hash that makes it binding — the
        # launcher refuses to run a deck whose sha256 does not match this line.
        header = text.split("\n[flow]")[0]
        (cdir / "prereg.md").write_text(
            f"# 사전등록 — {target.cell_id}\n\n"
            f"이 문서는 캠페인 덱 `{deck_path.name}`의 헤더 그대로이며, MASTER 호출 이전에 "
            f"쓰였다. 아래 sha256이 발사 스크립트의 게이트 값이다.\n\n"
            f"    sha256 {sha}\n\n"
            f"```\n{header}\n```\n", encoding="utf-8")
        self.ctx.update(deck_path=deck_path, deck_sha=sha, overrides=overrides)
        return {"deck": str(deck_path), "deck_sha256": sha, "overrides": overrides}

    def _do_arm_scripts(self, target: Target) -> dict[str, Any]:
        cfg, cdir = self.cfg, self.cfg.cell_dir(target.cell_id)
        pc = self.ctx["precheck"]
        cid, tag = target.cell_id, f"autoeng_{target.cell_id}"
        deck_name = f"{tag}.inp"
        champion, sha = pc["champion"], self.ctx["deck_sha"]

        (cdir / f"probe_{cid}.py").write_text(
            render_probe_script(cfg, target, pc["marks"], champion, deck_name), encoding="utf-8")
        (cdir / f"run_{tag}_probe.bat").write_text(
            render_run_bat(cfg, target, deck_name, f"{tag}_probe", f"probe_{cid}.py"),
            encoding="utf-8")
        (cdir / f"launch_{tag}_probe.ps1").write_text(
            render_launch_ps1(cfg, target, deck_name, f"{tag}_probe", sha, champion),
            encoding="utf-8")
        (cdir / f"run_{tag}.bat").write_text(
            render_run_bat(cfg, target, deck_name, tag,
                           f"-m lpopt optimize --input {deck_name} "
                           f"--run-dir runs/{tag} --no-early-stop"), encoding="utf-8")
        (cdir / f"launch_{tag}.ps1").write_text(
            render_launch_ps1(cfg, target, deck_name, tag, sha, champion,
                              fresh_run_dir=f"runs/{tag}"), encoding="utf-8")
        return {"scripts": sorted(p.name for p in cdir.glob("*"))}

    def _do_probe_readout(self, target: Target) -> dict[str, Any]:
        p = self.cfg.cell_dir(target.cell_id) / "blind_probe.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        out = {"n_probe": data.get("n_probe"), "n_converged": data.get("n_converged"),
               "per_target": data.get("per_target", {}), "model_dir": data.get("model_dir")}
        (self.cfg.cell_dir(target.cell_id) / "probe_readout.json").write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        return out

    def _do_store_backup(self, target: Target) -> dict[str, Any]:
        sd = self.cfg.p(self.cfg.store_dir)
        stamp = time.strftime("%Y%m%d")
        made = []
        for name in ("records.parquet", "maps.npz"):
            src = sd / name
            if src.exists():
                dst = sd / f"{name}.bak_pre_{target.cell_id}_{stamp}"
                shutil.copy2(src, dst)
                made.append(dst.name)
        return {"backups": made}

    def _do_mesh_baseline(self, target: Target) -> dict[str, Any]:
        md = self.cfg.p(self.cfg.mesh_dir)
        old = Path(self.ctx["precheck"]["champion"]).name
        made = []
        for name in ("model_bias.csv", "cell_verdicts.csv"):
            src = md / name
            if src.exists():
                dst = md / f"{src.stem}_{old}.csv"
                shutil.copy2(src, dst)
                made.append(dst.name)
        return {"preserved": made}

    def _do_retrain_prereg(self, target: Target) -> dict[str, Any]:
        """Pin the input hashes BEFORE the split is persisted (the S1x convention)."""
        arm = self.ctx.get("arm", "arm")
        pinned = {}
        for rel in (f"{self.cfg.store_dir}/records.parquet",
                    f"{self.cfg.store_dir}/maps.npz",
                    f"{self.cfg.store_dir}/fuel_types.parquet",
                    f"data/splits/{self.ctx.get('parent_split', 'S1')}.json"):
            p = self.cfg.p(rel)
            if p.exists():
                pinned[rel] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                               "bytes": p.stat().st_size}
        path = self.cfg.p(f"data/reports/ab2_addendum_{arm.upper()}_"
                          f"{time.strftime('%Y%m%d')}.md")
        lines = [f"# 재학습 라운드 사전등록 — {arm} (autoeng, cell {target.cell_id})", "",
                 f"작성 시각 {time.strftime('%Y-%m-%d %H:%M:%S')} — split 기록 **이전**.", "",
                 "## 입력 해시 (고정)", "", "| 항목 | sha256 | bytes |", "|---|---|---|"]
        for k, v in pinned.items():
            lines.append(f"| `{k}` | `{v['sha256']}` | {v['bytes']:,} |")
        lines += ["", f"## 절차", "",
                  f"- split: `build_split_S1b.py --parent {self.ctx.get('parent_split')} "
                  f"--name {self.ctx.get('split')} --holdout-new-campaigns`",
                  f"- 학습: 238 GPU1, s1f/s1g 레시피에서 `--split` 만 교체",
                  f"- 게이트: `lpopt gate-promote --check-only` -> PASS 시 승격",
                  "", "## 사전 약속", "",
                  "- 방향: 이 셀의 프론티어 라벨이 흡수되면 이웃 feed의 비관 편향이 **감소**해야 한다",
                  "  (comparison_readout.md 10.3의 82-86% 모델 기여 측정과 같은 방향).",
                  "- 게이트 FAIL은 기록된 폴백: 현 챔피언 유지 + 플래그. 사후에 기준을 바꾸지 않는다.",
                  ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return {"prereg": str(path), "pinned": list(pinned)}

    def _do_cell_report(self, target: Target) -> dict[str, Any]:
        cdir = self.cfg.cell_dir(target.cell_id)
        pc = self.ctx.get("precheck", {})
        marks = pc.get("marks", {})
        probe = self.state.result(target.cell_id, "probe_readout")
        L = [f"# autoeng 셀 보고서 — {target.cell_id}", "",
             f"생성 {time.strftime('%Y-%m-%d %H:%M:%S')} · 챔피언 {pc.get('champion')} · "
             f"덱 sha256 `{self.ctx.get('deck_sha', 'n/a')}`", "",
             "## 등록한 마크 대비", "", "| 마크 | 등록값 | 결과 |", "|---|---|---|"]
        for k, label in (("store_f_r_floor", "스토어 F_r 하한"),
                         ("db_f_r_min", "DB 진실값"),
                         ("mesh_min_pred_f_r", "모델 예측 하한"),
                         ("corrected_floor", "편향보정 하한")):
            v = marks.get(k)
            L.append(f"| {label} | {v if v is None else f'{v:.4f}'} | (캠페인 결과로 채움) |")
        L += ["", "## 블라인드 전이 프로브 (사전 실력)", ""]
        if probe.get("per_target"):
            L += ["| target | n | MAE | bias | spearman | cov1 |", "|---|---|---|---|---|---|"]
            for t, s in probe["per_target"].items():
                L.append(f"| {t} | {s.get('n')} | {s.get('mae'):.4g} | {s.get('bias'):+.4g} | "
                         f"{s.get('spearman'):.3f} | {s.get('cov1'):.2f} |")
        else:
            L.append("(프로브 판독 없음)")
        L += ["", "## 스텝 원장", "", "| 스텝 | 상태 | MASTER 콜 |", "|---|---|---|"]
        for ev in self.state.events:
            if ev.get("cell") == target.cell_id and ev["kind"] in ("step_done", "step_skip", "step_fail"):
                L.append(f"| {ev['step']} | {ev['kind'][5:]} | {ev.get('master_calls', 0)} |")
        (cdir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        self._append_notebook(target)
        return {"report": str(cdir / "report.md")}

    _INPROC = {
        "precheck": "_do_precheck", "prereg": "_do_prereg", "arm_scripts": "_do_arm_scripts",
        "probe_readout": "_do_probe_readout", "store_backup": "_do_store_backup",
        "mesh_baseline": "_do_mesh_baseline", "retrain_prereg": "_do_retrain_prereg",
        "cell_report": "_do_cell_report",
    }

    # -- lab notebook ------------------------------------------------------- #
    def _append_notebook(self, target: Target) -> None:
        path = self.cfg.p(self.cfg.notebook)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                "# AUTOENG_LOG — 자동 엔지니어 실험노트\n\n"
                "autoeng.py가 셀마다 추가로 기록한다. 사람이 읽는 쪽이 원본이고, "
                "기계가 읽는 원장은 `data/autoeng/<run_id>/state.jsonl`이다.\n", encoding="utf-8")
        pc = self.ctx.get("precheck", {})
        marks = pc.get("marks", {})
        spent = sum(int(e.get("master_calls", 0)) for e in self.state.events
                    if e.get("cell") == target.cell_id and e["kind"] == "step_done")
        fmt = lambda v, f="{:.4f}": ("n/a" if v is None else f.format(v))  # noqa: E731
        block = [
            "", "---", "",
            f"## {time.strftime('%Y-%m-%d %H:%M')} · {target.cell_id} "
            f"({target.library}, champion {pc.get('champion')})", "",
            f"- 사전 마크: 스토어 하한 {fmt(marks.get('store_f_r_floor'))} · "
            f"DB 진실값 {fmt(marks.get('db_f_r_min'))} · "
            f"모델 예측 하한 {fmt(marks.get('mesh_min_pred_f_r'))} "
            f"(판정 {marks.get('mesh_verdict', 'n/a')})",
            f"- 자산: level {pc.get('assets', {}).get('fallback_level')} "
            f"`{pc.get('assets', {}).get('restart_provenance')}`",
            f"- 엘리트 부모: {marks.get('elite_parents_feasible')}개 "
            f"(feed별 {marks.get('elite_parent_feeds')})",
            f"- 이 셀에서 쓴 MASTER 콜: {spent} / 누적 {self.state.master_calls()} "
            f"(상한 {self.cfg.master_budget_total})",
            f"- 셀 보고서: `{self.cfg.cell_dir(target.cell_id) / 'report.md'}`",
        ]
        for w in pc.get("warnings", []):
            block.append(f"- ⚠ {w}")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(block) + "\n")

    # -- step execution ----------------------------------------------------- #
    def execute_step(self, target: Target, step: Step) -> bool:
        # ``--dry-run`` promises "executes nothing".  Enforce it here rather than
        # relying on every caller to take the plan_only branch.
        if self.dry_run:
            raise RuntimeError(
                "autoeng is in --dry-run: execute_step must not be reached "
                f"(step {step.name!r} on {target.cell_id})"
            )
        cid = target.cell_id
        if self.state.step_status(cid, step.name) in ("done", "skipped"):
            self._log(f"  · {step.name}: 이미 완료 — 건너뜀 (resume)")
            return True
        if step.skip_if and self.ctx.get(step.skip_if):
            self.state.append("step_skip", cell=cid, step=step.name,
                              detail=f"skip_if={step.skip_if}")
            self._log(f"  · {step.name}: 조건부 생략 ({step.skip_if})")
            return True
        if step.master_calls and not self._check_budget(step.master_calls, cid):
            return False

        self.state.append("step_start", cell=cid, step=step.name, stage=step.stage,
                          where=step.where, argv=list(step.argv))
        self._log(f"  ▸ {step.name} [{step.where}] {step.what}")
        try:
            if step.where == "inproc":
                fn = getattr(self, self._INPROC[step.name])
                result = fn(target) or {}
            else:
                guard_argv(self.cfg, step.argv)
                r = self.runner(step.argv, cwd=self.cfg.root)
                if r.rc != 0:
                    raise RuntimeError(f"rc={r.rc}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                result = {"rc": r.rc, "stdout_tail": r.stdout[-1000:]}
                if step.poll_argv:
                    result["poll"] = self._poll(step)
        except Exception as exc:                                    # noqa: BLE001
            self.state.append("step_fail", cell=cid, step=step.name, error=str(exc)[:4000])
            self._log(f"  ✗ {step.name} 실패: {exc}")
            return False

        self.state.append("step_done", cell=cid, step=step.name,
                          master_calls=step.master_calls, result=result)
        if step.gate:
            return self._evaluate_gate(target, step, result)
        return True

    def _poll(self, step: Step) -> dict[str, Any]:
        t0, interval = time.time(), 60
        while time.time() - t0 < step.poll_timeout_s:
            r = self.runner(step.poll_argv, cwd=self.cfg.root)
            if step.poll_until in (r.stdout or ""):
                return {"waited_s": round(time.time() - t0, 1), "ok": True}
            time.sleep(interval)
        raise TimeoutError(f"{step.name}: poll timed out after {step.poll_timeout_s}s")

    def _evaluate_gate(self, target: Target, step: Step, result: dict) -> bool:
        gate = step.gate
        if gate == "new_assembly":
            self._pause(gate, target.cell_id,
                        "신규 집합체는 DeCART2D 설계 체인이 필요하다. v1은 사전점검까지만 하고 "
                        "사람에게 넘긴다 (paramA 선례: DeCART -> library -> bootstrap).")
            return False
        if gate == "retrain_promote_fail":
            arm = self.ctx.get("arm", "arm")
            gp = self.cfg.p(f"data/reports/gate_{arm}_checkonly.json")
            passed = False
            if gp.exists():
                passed = bool(json.loads(gp.read_text(encoding="utf-8")).get("pass"))
            self.ctx["gate_failed"] = not passed
            self.state.append("gate_result", cell=target.cell_id, step=gate, passed=passed)
            if passed:
                self._log("  ✓ 게이트 PASS — 승격 진행")
                return True
            detail = ("무회귀/legacy-tail 게이트 FAIL. 등록된 폴백은 '현 챔피언 유지 + 플래그'이며 "
                      "기준을 사후에 완화하지 않는다. 승인하면 현 챔피언으로 다음 셀을 계속한다.")
            if gate in self.cfg.pause_for_approval:
                self._pause(gate, target.cell_id, detail)
                return False
            self.state.append("note", cell=target.cell_id, detail=detail + " (게이트 미설정, 자동 계속)")
            return True
        return True

    # -- the loop ----------------------------------------------------------- #
    def run(self, *, max_cells: int | None = None) -> int:
        cfg = self.cfg
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        champion = read_champion(cfg)
        try:
            arm, split, parent_split = next_arm(champion)
        except ValueError as exc:
            arm = split = parent_split = ""
            self._log(f"[autoeng] 경고: {exc}")

        order = order_targets(cfg, cfg.targets, self._opened())
        self._log(f"[autoeng] run_id={cfg.run_id}  챔피언={champion}  "
                  f"셀 {len(order)}개  MASTER 누적 {self.state.master_calls()}/"
                  f"{cfg.master_budget_total}")
        self._log(f"[autoeng] 순서 (전이 인접성 -> DB 프론티어): "
                  f"{', '.join(t.cell_id for t in order)}")

        done = 0
        for target in order:
            if max_cells is not None and done >= max_cells:
                break
            if self.state.step_status(target.cell_id, "cell") == "done":
                continue
            self.ctx = {"arm": arm, "split": split, "parent_split": parent_split}
            self.state.append("cell_start", cell=target.cell_id)
            self._log(f"\n[autoeng] === {target.cell_id} ===")
            steps = plan_cell(cfg, target, champion=champion, arm=arm, split=split,
                              parent_split=parent_split)
            for step in steps:
                if not self.execute_step(target, step):
                    self._log(f"[autoeng] {target.cell_id}에서 중단 ({step.name}).")
                    return 10
            self.state.append("cell_done", cell=target.cell_id)
            done += 1
            # a promotion changes the champion for the NEXT cell.
            if not self.ctx.get("gate_failed") and arm:
                champion = f"data/models/{arm}"
                arm, split, parent_split = next_arm(champion)
        self._log(f"\n[autoeng] 완료: {done}개 셀, MASTER 누적 {self.state.master_calls()}.")
        return 0

    # -- dry run ------------------------------------------------------------ #
    def plan_only(self, *, max_cells: int | None = None) -> list[tuple[Target, list[Step]]]:
        """Build the full plan WITHOUT executing or writing anything."""

        cfg = self.cfg
        champion = read_champion(cfg)
        try:
            arm, split, parent_split = next_arm(champion)
        except ValueError:
            arm = split = parent_split = ""
        order = order_targets(cfg, cfg.targets, self._opened())
        if max_cells is not None:
            order = order[:max_cells]
        return [(t, plan_cell(cfg, t, champion=champion, arm=arm, split=split,
                              parent_split=parent_split)) for t in order]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_plan(cfg: AutoengConfig, plan: list[tuple[Target, list[Step]]],
                *, show_marks: bool = True) -> None:
    champion = read_champion(cfg)
    print("=" * 100)
    print(f"autoeng DRY RUN — run_id={cfg.run_id}  champion={champion}")
    print(f"  fleet: campaigns/probe/mesh -> {cfg.fleet.campaign_host}  "
          f"training -> 238 via {cfg.fleet.train_deck} (GPU 1)")
    print(f"  forbidden: {', '.join(cfg.fleet.forbidden)}")
    print(f"  human gates: {', '.join(cfg.pause_for_approval)}")
    print(f"  MASTER budget cap: {cfg.master_budget_total}")
    print("=" * 100)
    grand = 0
    for target, steps in plan:
        calls = sum(s.master_calls for s in steps)
        grand += calls
        print(f"\n### TARGET {target.cell_id}   library={target.library}   "
              f"probe={target.probe_budget} + open={target.budget} = {calls} MASTER calls")
        if show_marks and not target.is_new_assembly:
            try:
                pc = precheck(cfg, target)
                a, m = pc.get("assets", {}), pc.get("marks", {})
                print(f"  PRECHECK  assets: level {a.get('fallback_level')} "
                      f"{a.get('restart_provenance')}  resolvable={a.get('resolvable')}")
                print(f"            store : {m.get('store_rows')} rows / "
                      f"{m.get('store_converged')} converged / {m.get('store_feasible')} feasible"
                      f"   F_r floor {m.get('store_f_r_floor')}")
                print(f"            DB    : {m.get('db_n_cores')} cores  "
                      f"F_r_min {m.get('db_f_r_min')}  EFPD {m.get('db_best_efpd')}")
                print(f"            model : predicted floor {m.get('mesh_min_pred_f_r')}  "
                      f"corrected {m.get('corrected_floor')}  verdict {m.get('mesh_verdict')}")
                print(f"            derived: seed {m.get('random_seed')}  "
                      f"near_miss_f_r {m.get('near_miss_f_r')}  "
                      f"cycle_target {m.get('cycle_target_efpd')} "
                      f"+-{m.get('cycle_tolerance_efpd')}")
                for w in pc.get("warnings", []):
                    print(f"            ⚠ {w}")
                for b in pc.get("blockers", []):
                    print(f"            ⛔ {b}")
            except Exception as exc:                                # noqa: BLE001
                print(f"  PRECHECK failed: {exc}")
        print(f"  {len(steps)} steps:")
        for i, s in enumerate(steps, 1):
            call = f" [{s.master_calls} MASTER]" if s.master_calls else ""
            skip = f" (skip_if {s.skip_if})" if s.skip_if else ""
            gate = f" <GATE {s.gate}>" if s.gate else ""
            print(f"   {i:>2}. {s.stage:<11} {s.where:<7} {s.name:<16}{call}{skip}{gate}")
            print(f"       {s.what}")
            if s.argv:
                print(f"       $ {' '.join(s.argv)}")
            if s.poll_argv:
                print(f"       poll until {s.poll_until!r}: $ {' '.join(s.poll_argv)}")
            for w in s.writes:
                print(f"       -> {w}")
    print("\n" + "=" * 100)
    print(f"TOTAL: {sum(len(s) for _, s in plan)} steps across {len(plan)} cell(s), "
          f"{grand} MASTER calls planned "
          f"(cap {cfg.master_budget_total}).  NOTHING WAS EXECUTED.")
    print("=" * 100)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autoeng", description=__doc__.split("\n")[0])
    ap.add_argument("--config", "-c", default="autoeng.toml", help="autoeng target TOML")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the full plan; execute and write nothing")
    ap.add_argument("--resume", action="store_true", help="continue from the state log")
    ap.add_argument("--status", action="store_true", help="print the state log ledger and exit")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--no-marks", action="store_true",
                    help="dry-run without the PRECHECK measurements (faster, less useful)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    cfg = load_autoeng_config(args.config)
    eng = AutoEngineer(cfg, dry_run=args.dry_run)

    if args.status:
        print(f"state: {cfg.state_path}")
        for ev in eng.state.events:
            print(f"  [{ev['iso']}] {ev['kind']:<11} {ev.get('cell',''):<14} "
                  f"{ev.get('step','')}  {ev.get('detail','')}")
        print(f"MASTER calls accounted: {eng.state.master_calls()}")
        return 0

    if args.dry_run:
        _print_plan(cfg, eng.plan_only(max_cells=args.max_cells), show_marks=not args.no_marks)
        return 0

    return eng.run(max_cells=args.max_cells)


if __name__ == "__main__":
    raise SystemExit(main())
