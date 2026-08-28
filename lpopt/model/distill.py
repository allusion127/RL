"""Soft-target distillation from per-cell best historical teachers (v5 arm D).

Recorded decision (2026-07-19, memory ``lpopt-project.md``): at the 36-cell
curriculum completion, run ONE pre-registered A/B of a plain full retrain against
a soft-target distilled retrain.  The distillation is deliberately conservative —
the forensic that motivated it measured a recovery gap of ~0 (mean +0.02, and 8
of 12 cells peaked at a LATER champion), i.e. there is no catastrophic
forgetting to repair — so the aggressive variants were REJECTED:

* **No row selection.** The student trains on the FULL corpus, exactly as the
  baseline arm does.  Distillation only adds a soft-target term; it never
  subsets, reweights or drops rows.
* **The 5-member ensemble is preserved.** Distillation is per-member (each
  student member sees the same soft targets), not a collapse to one model.
* **Pin-BU targets are excluded.** ``max_pin_burnup`` (and, when promoted,
  ``max_assembly_burnup``) are dropped from the soft-target mask: Dataset A's pin
  label is a MOCHA-cache surrogate (data/reports/pinbu_forensics.md), so a
  teacher's pin head would distil a known artifact into the student.
* **Per-cell best teacher.** Each row's soft target comes from the historical
  champion that scored best on that row's ``(feed, e_core-bin)`` cell — not from
  a single global teacher — which is what makes the ensemble-of-history usable
  without keeping every checkpoint at serve time.

This module only BUILDS the soft-target cache (an ``.npz`` keyed by record_id).
``train.py`` consumes it behind ``--distill-targets``; with no cache the training
path is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: Targets NEVER distilled (see the module docstring): the pin/assembly burnup
#: axes whose historical labels are cache surrogates rather than pin calcs.
EXCLUDED_TARGETS: frozenset[str] = frozenset({
    "max_pin_burnup", "max_assembly_burnup",
})
#: filename of the built cache.
DISTILL_CACHE_NAME = "distill_soft_targets.npz"
DISTILL_SCHEMA = "distill_soft_targets_v1"


def teacher_mask(target_names: Sequence[str]) -> np.ndarray:
    """``float32[T]`` mask: 1 for a distillable target, 0 for an excluded one."""
    return np.asarray(
        [0.0 if n in EXCLUDED_TARGETS else 1.0 for n in target_names],
        dtype=np.float32,
    )


def load_teacher_map(path: str | Path) -> dict[str, str]:
    """Load the ``{cell_key: teacher_model_dir}`` map.

    Accepts either a bare mapping or ``{"teachers": {...}}``; every value must be
    a directory that exists and holds at least one ``member_*`` checkpoint (the
    A/B runner validates this in ``--dry-run`` so a typo fails before the GPU is
    reserved, not four hours in).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    teachers = raw.get("teachers", raw) if isinstance(raw, dict) else {}
    return {str(k): str(v) for k, v in teachers.items()}


def validate_teacher_map(teachers: Mapping[str, str]) -> list[str]:
    """Return a list of human-readable problems with a teacher map (empty == ok)."""
    problems: list[str] = []
    if not teachers:
        problems.append("teacher map is empty")
    for cell, d in sorted(teachers.items()):
        p = Path(d)
        if not p.is_dir():
            problems.append(f"cell {cell}: teacher dir does not exist: {d}")
        elif not sorted(p.glob("member_*")):
            problems.append(f"cell {cell}: no member_* checkpoints under {d}")
    return problems


def build_soft_targets(
    df: Any,
    teachers: Mapping[str, str],
    *,
    cell_keys: Sequence[str],
    target_names: Sequence[str],
    store_dir: str | Path = "data/store",
    device: str = "cpu",
    library_id: str = "ga80",
    batch_size: int = 1024,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score every row with its cell's teacher; return (and optionally cache) the
    ``[N, T]`` soft-target matrix in RAW target units plus its validity mask.

    A row whose cell has no teacher (or whose teacher errors) gets an all-zero
    mask row and contributes nothing to the distillation loss — the student then
    simply trains on the hard label there, which is the desired graceful
    degradation for a cell the history never covered.
    """
    from .model_api import PosValCnnBackend

    n = len(df)
    t = len(target_names)
    soft = np.full((n, t), np.nan, dtype=np.float32)
    mask = np.zeros((n, t), dtype=np.float32)
    base_mask = teacher_mask(target_names)

    keys = np.asarray([str(k) for k in cell_keys])
    teacher_cells = {str(c) for c in teachers}
    # rows whose cell HAS a teacher: the denominator the fail-loud guard measures
    # match against.  A row outside every teacher cell (a legacy row) is not
    # "intended" for a soft target and its absence is not a defect.
    n_intended = int(np.isin(keys, list(teacher_cells)).sum())
    for cell, model_dir in sorted(teachers.items()):
        sel = np.flatnonzero(keys == str(cell))
        if not sel.size:
            continue
        try:
            backend = PosValCnnBackend.from_dir(
                model_dir, store_dir=store_dir, library_id=library_id,
                device=device)
        except Exception as exc:      # noqa: BLE001 - a bad teacher must not abort
            print(f"WARNING: distill teacher for cell {cell} unusable: {exc}",
                  flush=True)
            continue
        t_names = list(backend.target_names)
        cols = [t_names.index(nm) if nm in t_names else None for nm in target_names]
        for start in range(0, sel.size, int(batch_size)):
            idx = sel[start:start + int(batch_size)]
            rows = df.iloc[idx]
            # predict_rows_raw honours each row's OWN library provenance and
            # returns the 7-column surrogate layout; map it back to dataset order.
            sur = np.asarray(backend.predict_rows_raw(rows), dtype=float)
            from .model_api import _TARGET_TO_SURROGATE_COL
            for k, nm in enumerate(target_names):
                if cols[k] is None or base_mask[k] == 0.0:
                    continue
                scol = _TARGET_TO_SURROGATE_COL.get(nm)
                if scol is None:
                    continue
                vals = sur[:, scol]
                soft[idx, k] = vals
                mask[idx, k] = np.isfinite(vals).astype(np.float32)

    n_soft = int((mask.sum(axis=1) > 0).sum())
    # Fail loud at BUILD: if rows WERE intended (their cell has a teacher) but
    # NONE got a soft target, the cell-key join is broken (the exact defect that
    # produced the 0/42867 decoy — teacher cells keyed one way, row cells another).
    # A silent all-zero cache would train a distill arm identical to the baseline.
    if n_intended > 0 and n_soft == 0:
        raise ValueError(
            f"distillation soft-target build matched 0 of {n_intended} intended "
            f"rows: the row cell-keys do not match ANY of the {len(teacher_cells)} "
            f"teacher cells (e.g. rows={sorted(set(keys))[:2]} vs "
            f"teachers={sorted(teacher_cells)[:2]}). The keys must be built by the "
            f"same recipe on both sides (campaign / cell_id).")
    if n_soft == 0:
        raise ValueError(
            "distillation soft-target build produced no soft targets at all "
            "(no train row fell in a teacher cell); check the teacher map.")
    artifact = {
        "schema": DISTILL_SCHEMA,
        "record_ids": np.asarray(df["record_id"].astype(str).tolist(), dtype=object),
        "soft": soft,
        "mask": mask,
        "target_names": list(target_names),
        "n_cells": len(teachers),
        "n_intended": n_intended,
        "n_soft": n_soft,
    }
    print(f"=== distill build: {n_soft} soft-target rows of {n_intended} intended "
          f"({len(teacher_cells)} teacher cells, {n} corpus rows) ===", flush=True)
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            record_ids=artifact["record_ids"].astype(str),
            soft=soft, mask=mask,
            target_names=np.asarray(list(target_names), dtype=str),
            schema=DISTILL_SCHEMA,
            n_intended=np.asarray(n_intended),
            n_soft=np.asarray(n_soft),
        )
    return artifact


def load_soft_targets(path: str | Path) -> dict[str, Any]:
    """Load a built cache; returns ``{record_ids, soft, mask, target_names, ...}``."""
    z = np.load(Path(path), allow_pickle=False)
    out = {
        "record_ids": [str(x) for x in z["record_ids"]],
        "soft": np.asarray(z["soft"], dtype=np.float32),
        "mask": np.asarray(z["mask"], dtype=np.float32),
        "target_names": [str(x) for x in z["target_names"]],
    }
    # ``n_soft`` is the count of intended rows the BUILD matched; the attach guard
    # measures how many of them survived the record_id join on the remote store.
    # Recompute from the mask if an older cache lacks the field (version-robust).
    out["n_soft"] = (int(z["n_soft"]) if "n_soft" in z.files
                     else int((out["mask"].sum(axis=1) > 0).sum()))
    out["n_intended"] = int(z["n_intended"]) if "n_intended" in z.files else out["n_soft"]
    return out


def align_soft_targets(cache: Mapping[str, Any], record_ids: Sequence[str],
                       target_names: Sequence[str]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Reindex a cache onto ``record_ids`` x ``target_names``.

    Rows / targets the cache does not carry get a zero mask, so a cache built
    against an older target inventory still works for the columns it shares.
    """
    n, t = len(record_ids), len(target_names)
    soft = np.zeros((n, t), dtype=np.float32)
    mask = np.zeros((n, t), dtype=np.float32)
    pos = {rid: i for i, rid in enumerate(cache["record_ids"])}
    tpos = {nm: k for k, nm in enumerate(cache["target_names"])}
    cols = [tpos.get(nm) for nm in target_names]
    for i, rid in enumerate(record_ids):
        j = pos.get(str(rid))
        if j is None:
            continue
        for k, c in enumerate(cols):
            if c is None or target_names[k] in EXCLUDED_TARGETS:
                continue
            soft[i, k] = cache["soft"][j, c]
            mask[i, k] = cache["mask"][j, c]
    soft = np.nan_to_num(soft, nan=0.0)
    return soft, mask


__all__ = [
    "DISTILL_CACHE_NAME",
    "DISTILL_SCHEMA",
    "EXCLUDED_TARGETS",
    "align_soft_targets",
    "build_soft_targets",
    "load_soft_targets",
    "load_teacher_map",
    "teacher_mask",
    "validate_teacher_map",
]
