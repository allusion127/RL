"""The C2 judging slice and the split-freshness refusal — program sections 7.1 / 7.2.

C2 is not "fold C".  Section 7.1 defines it as fold C **after refinement**, and
the refinement is the whole point: fold C's set-difference predicate proves a row
was produced after the split was frozen, but it does not prove the row is
*independent* of the training set.  Three measured leaks survive it:

======================================  ==============================================
leak                                    section 7.1's measurement
======================================  ==============================================
lineage                                 236 of 2,004 fold-C map rows (11.8%) are ONE
                                        mutation away from a training row
symmetry                                transpose pairs are physically the same core
                                        (RMS 0.00285, rho 0.99996); 24 of 188
                                        ``case_pair`` groups straddle the split
provenance                              33.1% of fold-C map rows -- 36.6% in the flat
                                        tercile -- were proposed by the incumbent
======================================  ==============================================

The first two are removed.  The third cannot be removed (it would empty the
slice), so it is **stratified**: :class:`C2Slice` carries the production-only
subset and section 8.3 requires the primary result to hold there too.

Split freshness is a refusal, not a warning
-------------------------------------------
``data/splits/S2.json`` on disk is the LEGACY leave-pair-out manifest
(``kind="leave_pair"``, holdout pairs ``C3_C6``/``A01_B05``) written 2026-07-18.
It is syntactically perfect: status ``ok``, every id present in the store, folds
that compute without error.  That is exactly what makes it dangerous -- a
flatness A/B pointed at it would produce a full, plausible, *wrong* verdict,
because its "fold C" is simply every row added to the store since July 18
(13,594 rows, 24.9% of the store) rather than the pre-registered C2 slice.

So :func:`require_program_split` **raises**.  A missing artifact announces
itself; a stale one has to be made to.  The audit that accompanies the refusal
carries the measured numbers so the reader can see *why* rather than being told.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .folds import MIN_CELL_ROWS, cell_key, fold_frame, proposal_mask
from .splits import SplitManifest

#: A manifest must declare this to be usable as the flatness program's split.
FLATNESS_PROGRAM = "flatness_first"
#: Schema tag written into every audit / slice provenance block.
AUDIT_SCHEMA = "split_audit_v1"
C2_SCHEMA = "c2_slice_v1"
#: A manifest that does not cover this fraction of the store is stale: its
#: "everything not in the manifest" fold has stopped being a held-out slice and
#: become "the store's recent growth".
MAX_UNCOVERED_FRAC = 0.05
#: The command that fixes it (quoted in the refusal).
REGEN_CMD = "python -m lpopt.tools.audit_c2_split --regenerate"


class SplitStaleError(RuntimeError):
    """Raised when a judging slice is asked for on a split that cannot carry it."""


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def store_fingerprint(df: pd.DataFrame) -> str:
    """Stable digest of the store's record-id set (order-independent)."""
    ids = sorted(str(x) for x in df["record_id"].tolist())
    h = hashlib.sha1()
    h.update(str(len(ids)).encode())
    for i in ids:
        h.update(i.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _manifest_ids(manifest: SplitManifest) -> set[str]:
    return {str(i) for i in manifest.record_ids("all")}


# --------------------------------------------------------------------------- #
# freshness audit
# --------------------------------------------------------------------------- #
def audit_split(manifest: SplitManifest, df: pd.DataFrame, *,
                program: str = FLATNESS_PROGRAM,
                max_uncovered_frac: float = MAX_UNCOVERED_FRAC) -> dict[str, Any]:
    """Measure whether ``manifest`` can still carry a judging slice.

    Returns a provenance block with ``stale`` and an itemized ``reasons`` list.
    Pure: it reads, measures and reports; it never mutates anything.
    """
    ids = _manifest_ids(manifest)
    store_ids = {str(x) for x in df["record_id"].tolist()}
    missing = sorted(ids - store_ids)
    uncovered = sorted(store_ids - ids)
    frac = (len(uncovered) / len(store_ids)) if store_ids else 0.0
    declared = str((manifest.predicate or {}).get("program", ""))
    declared_fp = str((manifest.groups or {}).get("store_fingerprint", ""))
    fp = store_fingerprint(df)

    reasons: list[str] = []
    if declared != program:
        reasons.append(
            f"manifest declares program={declared or '(none)'!r}, not {program!r}: "
            f"this is the {manifest.kind!r} split '{manifest.name}', which was "
            "built for a different question and whose held-out fold is not the "
            "pre-registered C2 slice")
    if manifest.status != "ok":
        reasons.append(f"manifest status is {manifest.status!r}")
    if missing:
        reasons.append(f"{len(missing)} manifest ids are absent from the store "
                       "(the store was rebuilt or pruned under the manifest)")
    if frac > float(max_uncovered_frac):
        reasons.append(
            f"{len(uncovered)} of {len(store_ids)} store rows ({100*frac:.1f}%) "
            f"are outside the manifest, over the {100*max_uncovered_frac:.0f}% "
            "bar: the complement fold is the store's growth since the split was "
            "written, not a designed holdout")
    if declared_fp and declared_fp != fp:
        reasons.append(f"store fingerprint {fp} != the {declared_fp} the manifest "
                       "was built against")

    return {
        "schema": AUDIT_SCHEMA,
        "split": manifest.name,
        "kind": manifest.kind,
        "status": manifest.status,
        "declared_program": declared or None,
        "required_program": program,
        "n_manifest_ids": len(ids),
        "n_train": manifest.n_train,
        "n_val": manifest.n_val,
        "n_store_rows": len(store_ids),
        "n_manifest_ids_missing_from_store": len(missing),
        "n_store_rows_outside_manifest": len(uncovered),
        "uncovered_frac": round(frac, 6),
        "max_uncovered_frac": float(max_uncovered_frac),
        "store_fingerprint": fp,
        "manifest_store_fingerprint": declared_fp or None,
        "stale": bool(reasons),
        "reasons": reasons,
        "remedy": REGEN_CMD,
    }


def render_audit(audit: Mapping[str, Any]) -> str:
    """Human-readable audit -- the body of the refusal message."""
    L = [f"split {audit.get('split')!r} (kind={audit.get('kind')}, "
         f"status={audit.get('status')}): "
         f"{'STALE' if audit.get('stale') else 'fresh'}",
         f"  declared program : {audit.get('declared_program')} "
         f"(required: {audit.get('required_program')})",
         f"  manifest ids     : {audit.get('n_manifest_ids')} "
         f"(train {audit.get('n_train')} / val {audit.get('n_val')})",
         f"  store rows       : {audit.get('n_store_rows')} "
         f"[fingerprint {audit.get('store_fingerprint')}]",
         f"  outside manifest : {audit.get('n_store_rows_outside_manifest')} "
         f"({100 * float(audit.get('uncovered_frac') or 0):.1f}%)",
         f"  missing from store: {audit.get('n_manifest_ids_missing_from_store')}"]
    for r in audit.get("reasons") or []:
        L.append(f"  ! {r}")
    if audit.get("stale"):
        L.append(f"  remedy: {audit.get('remedy')}")
    return "\n".join(L)


def require_program_split(manifest: SplitManifest, df: pd.DataFrame, *,
                          program: str = FLATNESS_PROGRAM,
                          allow_stale: bool = False,
                          max_uncovered_frac: float = MAX_UNCOVERED_FRAC
                          ) -> dict[str, Any]:
    """Return the audit, or raise :class:`SplitStaleError` if it is stale.

    ``allow_stale=True`` is provided for reproducing a historical scoring run.
    It is a keyword, it is never the default, and the returned audit still says
    ``stale: true`` -- so an artifact produced that way carries the admission
    with it.
    """
    audit = audit_split(manifest, df, program=program,
                        max_uncovered_frac=max_uncovered_frac)
    if audit["stale"] and not allow_stale:
        raise SplitStaleError(
            "refusing to build a judging slice on a stale split.\n"
            + render_audit(audit)
            + "\n(pass allow_stale=True only to reproduce a historical run; the "
              "audit will still record the slice as stale)")
    return audit


def write_audit(audit: Mapping[str, Any], path: str | Path) -> Path:
    """Persist an audit next to the manifest it describes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(audit), indent=1, ensure_ascii=False),
                 encoding="utf-8")
    return p


def mark_stale(split_path: str | Path, audit: Mapping[str, Any]) -> SplitManifest:
    """Stamp the staleness verdict INTO the manifest, non-destructively.

    ``train_ids``/``val_ids`` are untouched, so every existing reader keeps
    working; what changes is that ``status`` stops saying ``ok`` and
    ``groups.staleness_audit`` carries the measured provenance.  A manifest that
    has been invalidated should not be able to look valid at a glance -- that was
    the failure mode.
    """
    p = Path(split_path)
    m = SplitManifest.from_json(p)
    m.status = "stale"
    m.groups = dict(m.groups or {})
    m.groups["staleness_audit"] = dict(audit)
    m.to_json(p)
    return m


# --------------------------------------------------------------------------- #
# regeneration: the flatness program's own split (section 7.2)
# --------------------------------------------------------------------------- #
def make_flat_cell_split(df: pd.DataFrame, *, name: str = "S2", seed: int = 0,
                         e_core_holdout_from: float | None = 5.75,
                         cell_val_frac: float = 0.20,
                         min_cell_rows: int = MIN_CELL_ROWS,
                         program: str = FLATNESS_PROGRAM) -> SplitManifest:
    """CELL-level holdout split for the flatness program (section 7.2).

    Section 7.2 dissolves the draft's catch-22 (train on the high-``e_core`` rows
    and you cannot verify the fix; hold them out and you judge arms on a deficit
    the loss was never allowed to touch) by moving the holdout unit from the ROW
    to the CELL: whole high-``e_core`` cells are held out as the extrapolation
    stratum, the rest of that region trains, and the primary metrics are
    pre-registered on the in-domain stratum with the extrapolation stratum
    reported separately as a domain-shift diagnostic.

    Cell membership is a stable hash of the cell key, so adding rows to an
    existing cell never moves it between folds -- growth invariance, the same
    property :func:`~.splits.make_curriculum_split` relies on.
    """
    ids = sorted(str(x) for x in df["record_id"].tolist())
    cells = cell_key(df).to_numpy()
    e = pd.to_numeric(df.get("e_core"), errors="coerce").to_numpy(dtype=float)

    counts: dict[str, int] = {}
    cell_e: dict[str, list[float]] = {}
    for i, c in enumerate(cells):
        c = str(c)
        if not c:
            continue
        counts[c] = counts.get(c, 0) + 1
        if math.isfinite(e[i]):
            cell_e.setdefault(c, []).append(float(e[i]))

    extrapolation: list[str] = []
    in_domain: list[str] = []
    for c in sorted(counts):
        if counts[c] < min_cell_rows:
            continue
        med = float(np.median(cell_e[c])) if cell_e.get(c) else float("nan")
        if (e_core_holdout_from is not None and math.isfinite(med)
                and med >= float(e_core_holdout_from)):
            extrapolation.append(c)
        else:
            in_domain.append(c)

    def _h(text: str) -> float:
        return int(hashlib.sha1(f"{name}:{seed}:{text}".encode()).hexdigest()[:16],
                   16) / float(1 << 64)

    val_cells = sorted(c for c in in_domain if _h(c) < float(cell_val_frac))
    if not val_cells and in_domain:            # never emit an empty judging fold
        val_cells = [min(in_domain, key=lambda c: (_h(c), c))]
    holdout = set(val_cells) | set(extrapolation)

    cell_of = {str(r): str(c) for r, c in zip(df["record_id"].astype(str), cells)}
    val_ids = [i for i in ids if cell_of.get(i, "") in holdout]
    val_set = set(val_ids)
    train_ids = [i for i in ids if i not in val_set]
    return SplitManifest(
        name=name, kind="flat_cell_holdout", seed=int(seed),
        train_ids=train_ids, val_ids=val_ids, status="ok",
        predicate={"program": program, "kind": "cell_holdout",
                   "cell_val_frac": float(cell_val_frac),
                   "e_core_holdout_from": e_core_holdout_from,
                   "min_cell_rows": int(min_cell_rows)},
        groups={"store_fingerprint": store_fingerprint(df),
                "in_domain_val_cells": val_cells,
                "extrapolation_cells": sorted(extrapolation),
                "n_cells_total": len(counts)},
    )


# --------------------------------------------------------------------------- #
# the slice
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class C2Slice:
    """The judging rows after the section 7.1 refinements, with their provenance.

    ``provenance['source']`` says which fold they came from: ``val`` for a
    program split's designed cell-level holdout, ``foldC`` for a legacy split's
    "produced after the manifest was frozen" complement.
    """

    df: pd.DataFrame
    cells: np.ndarray
    is_proposal: np.ndarray
    frozen_cells: tuple[str, ...]
    provenance: dict[str, Any]
    audit: dict[str, Any]

    def __len__(self) -> int:
        return len(self.df)

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(self.df["record_id"].astype(str).tolist())

    @property
    def n_cells(self) -> int:
        return len(self.frozen_cells)

    def production_mask(self) -> np.ndarray:
        """Rows NOT proposed by a model (the stratum section 8.3 also requires)."""
        return ~self.is_proposal

    def truth(self, target: str) -> np.ndarray:
        if target not in self.df.columns:
            return np.full(len(self.df), np.nan)
        return pd.to_numeric(self.df[target], errors="coerce").to_numpy(dtype=float)


def _ancestor_chain(rid: str, parent_of: Mapping[str, str],
                    limit: int = 64) -> list[str]:
    out: list[str] = []
    seen = {rid}
    cur = parent_of.get(rid, "")
    while cur and cur not in seen and len(out) < limit:
        out.append(cur)
        seen.add(cur)
        cur = parent_of.get(cur, "")
    return out


def build_c2(df: pd.DataFrame, manifest: SplitManifest, *,
             require_targets: Sequence[str] = ("node_peak", "map_cov"),
             min_cell_rows: int = MIN_CELL_ROWS,
             program: str = FLATNESS_PROGRAM,
             allow_stale: bool = False,
             max_uncovered_frac: float = MAX_UNCOVERED_FRAC) -> C2Slice:
    """Build the frozen C2 judging slice, or refuse.

    The refusal comes first, on purpose: :func:`require_program_split` runs before
    a single row is selected, so a stale manifest cannot produce a slice at all.
    """
    audit = require_program_split(manifest, df, program=program,
                                  allow_stale=allow_stale,
                                  max_uncovered_frac=max_uncovered_frac)

    # Where the judging rows come from depends on what kind of split this is,
    # and getting it wrong is silent.  A PROGRAM split (section 7.2) covers every
    # store row and designates its holdout explicitly, so the judging slice is
    # its ``val`` fold and only ``train`` is contaminating.  A legacy split
    # designates nothing, so the only defensible slice is fold C -- rows in
    # NEITHER list, which the frozen manifest proves were produced afterwards --
    # and both lists contaminate.
    declared = str((manifest.predicate or {}).get("program", ""))
    if declared == program:
        source = "val"
        keep_ids = {str(i) for i in manifest.record_ids("val")}
        base = df.loc[df["record_id"].astype(str).isin(keep_ids)]
        if "converged" in base.columns:
            base = base.loc[base["converged"].fillna(False).astype(bool)]
        base = base.reset_index(drop=True)
        contaminated = {str(i) for i in manifest.record_ids("train")}
    else:
        source = "foldC"
        base = fold_frame(df, manifest, "C").df.reset_index(drop=True)
        contaminated = _manifest_ids(manifest)
    n_source = len(base)

    # Lineage edges over the WHOLE store, not just the slice: an ancestor may
    # itself be a row we are about to drop, and the chain has to reach past it.
    parent_of: dict[str, str] = {}
    if "parent_record_id" in df.columns:
        for rid, par in zip(df["record_id"].astype(str), df["parent_record_id"]):
            if par is None or (isinstance(par, float) and math.isnan(par)):
                continue
            text = str(par)
            if text in ("", "None", "nan", "NaT"):
                continue
            parent_of[str(rid)] = text

    keep = np.ones(len(base), dtype=bool)
    dropped = {"lineage": 0, "symmetry_pair": 0, "duplicate_pattern": 0,
               "missing_label": 0}

    # -- 1. transitive lineage ---------------------------------------------- #
    rids = base["record_id"].astype(str).tolist()
    lineage_bad = np.zeros(len(base), dtype=bool)
    for i, rid in enumerate(rids):
        for anc in _ancestor_chain(rid, parent_of):
            if anc in contaminated:
                lineage_bad[i] = True
                break
    dropped["lineage"] = int(lineage_bad.sum())
    keep &= ~lineage_bad

    # -- 2. symmetry / repeat pairs straddling the split --------------------- #
    sym_bad = np.zeros(len(base), dtype=bool)
    if "case_pair" in df.columns:
        tainted_pairs = {
            str(p) for p, r in zip(df["case_pair"].astype(str),
                                   df["record_id"].astype(str))
            if str(p) not in ("", "None", "nan") and r in contaminated}
        if tainted_pairs:
            sym_bad = base["case_pair"].astype(str).isin(tainted_pairs).to_numpy()
    dropped["symmetry_pair"] = int((sym_bad & keep).sum())
    keep &= ~sym_bad

    # -- 3. an identical pattern already seen in train/val ------------------- #
    dup_bad = np.zeros(len(base), dtype=bool)
    if "pattern" in df.columns:
        seen_patterns = {
            str(p) for p, r in zip(df["pattern"].astype(str),
                                   df["record_id"].astype(str))
            if r in contaminated and str(p) not in ("", "None", "nan")}
        if seen_patterns:
            dup_bad = base["pattern"].astype(str).isin(seen_patterns).to_numpy()
    dropped["duplicate_pattern"] = int((dup_bad & keep).sum())
    keep &= ~dup_bad

    # -- 4. rows carrying no flatness label are not part of a flatness slice - #
    label_bad = np.zeros(len(base), dtype=bool)
    for tgt in require_targets:
        if tgt in base.columns:
            label_bad |= ~pd.to_numeric(base[tgt], errors="coerce").notna().to_numpy()
        else:
            label_bad |= np.ones(len(base), dtype=bool)
    dropped["missing_label"] = int((label_bad & keep).sum())
    keep &= ~label_bad

    sub = base.loc[keep].reset_index(drop=True)
    cells = cell_key(sub).to_numpy()
    is_prop = proposal_mask(sub)

    counts: dict[str, int] = {}
    for c in cells:
        c = str(c)
        if c:
            counts[c] = counts.get(c, 0) + 1
    frozen = tuple(sorted(c for c, n in counts.items() if n >= min_cell_rows))

    in_frozen = np.asarray([str(c) in set(frozen) for c in cells], dtype=bool)
    n_prod = int((~is_prop & in_frozen).sum())
    h = hashlib.sha1()
    for c in frozen:
        h.update(c.encode())
        h.update(b"\x00")
    for r in sorted(sub.loc[in_frozen, "record_id"].astype(str).tolist()):
        h.update(r.encode())
        h.update(b"\x00")

    provenance = {
        "schema": C2_SCHEMA,
        "split": manifest.name,
        "source": source,
        "n_source": n_source,
        "n_fold_c": (n_source if source == "foldC" else None),
        "n_c2": int(len(sub)),
        "n_in_frozen_cells": int(in_frozen.sum()),
        "dropped": dropped,
        "n_cells_frozen": len(frozen),
        "min_cell_rows": int(min_cell_rows),
        "frozen_cells": list(frozen),
        "cell_manifest_sha1": h.hexdigest()[:16],
        "n_proposed": int((is_prop & in_frozen).sum()),
        "n_production": n_prod,
        "production_frac": (n_prod / int(in_frozen.sum())) if in_frozen.any() else 0.0,
        "require_targets": list(require_targets),
        "store_fingerprint": audit.get("store_fingerprint"),
        "stale_split": bool(audit.get("stale")),
    }
    return C2Slice(df=sub, cells=cells, is_proposal=is_prop, frozen_cells=frozen,
                   provenance=provenance, audit=dict(audit))


__all__ = [
    "AUDIT_SCHEMA", "C2_SCHEMA", "C2Slice", "FLATNESS_PROGRAM",
    "MAX_UNCOVERED_FRAC", "REGEN_CMD", "SplitStaleError", "audit_split",
    "build_c2", "make_flat_cell_split", "mark_stale", "render_audit",
    "require_program_split", "store_fingerprint", "write_audit",
]
