"""**prereg delta D** — the v3.1 out-of-fold metric module (§5d / §6).

This is the module ``lpopt.policy.train_v3.assert_v3_path_untouched`` names as
the thing that unblocks ``--stage2 on``.  What it owns, and nothing else:

* §5d — the four FITTED baselines (``random`` / ``class_freq`` / ``periph`` /
  ``gd_rule``) are refitted **inside each cross-fit block's own train fold** and
  the out-of-fold score column is stitched from the K fits, so no baseline ever
  scores a row it was fitted on.  The fitting code is not re-implemented here:
  :func:`~.train_v3.baseline_scores_v3` and :func:`~.train_v3.gd_rule_sign` are
  called, which is the same code the v3 results were produced with.
  ``policy_v2`` is NOT fitted — it is read from the blind CSV of §3b.
* §6a — clause 1 (parent-blocked AUC), clause 2A (NDCG@4-of-8 against the four
  fitted baselines), clause 2B (non-inferiority to ``policy_v2``, margin
  ``delta = 0.05``, one-sided 95%), clause 3 (within-cell, with the registered
  eligibility rule and its permutation test), clause 4 (served Platt spread).
  Clause 5 lives in :func:`~.train_v3.assert_stage2_init_is_stage1` because it
  is a structural property of the checkpoint, not of a fold.
* §5a — every ranking statistic is fed the REGISTERED gain ``y_fxy`` and
  :func:`assert_registered_gain` raises if a caller hands it raw ``-d_f_xy``.
  That refusal is the whole of v3 deviation §1.6, closed from this side.
* §3c-(4) — the measured permutation-null MDE per cell, which the prereg's
  table filled with the closed form ``0.55/sqrt(n_live)`` for all but three
  cells.

**ECE is REPORTED and NOT GATED.**  §5c registers the reason in advance: both
measured configurations (0.0521 raw, 0.0858 val-Platt) are already above 0.05,
so gating it would be registering a FAIL that is already known.  It is computed
here, printed next to Brier, and carried in the report under
``calibration`` with an explicit ``gated: false``.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .train import calibration, parent_blocked_auc
from .train_v2 import _delta, _paired_parent_bootstrap, _per_parent_blocked
from .train_v3 import (
    BASELINES, NDCG_K, REGRET_MIN_CANDIDATES, baseline_scores_v3,
    gd_rule_sign, ndcg_at_k,
)
from .v3 import (
    EVAL_LABEL_V3, MIN_POOL_LIVE_V31, PROSPECTIVE_CELL_V31, fit_platt_v31,
    platt_serve, targets_v3, xfit_indices,
)

# --------------------------------------------------------------------------- #
# registered constants — the bars this module judges against
# --------------------------------------------------------------------------- #
#: The frozen STEP 0-a artefacts (freeze stamp §S0.2 / §S0.5).  The refusal in
#: ``train_v3`` lifts only against THESE bytes: a metric module that will happily
#: score any assignment is not a pre-registration, and the whole point of
#: emitting ``splits_v31.csv`` as a file was that the assignment the gate is
#: computed on is hashable rather than re-derived.
SPLITS_V31_SHA256 = (
    "a62c9937e55ad47cf42b9a09c8d5a220efa8cdccacf03409017d52fd9c951e34")
STEPS_V31_SHA256 = (
    "ed74a6b4cc68683075c4bf9304fd6ad32c11de49eb3d6e41be7a6730ebf589ad")

#: §5d: the four baselines that are REFITTED inside each block.  ``policy_v2``
#: is the fifth registered baseline and is deliberately not in this tuple — it
#: is a frozen checkpoint's blind CSV and refitting it would not mean anything.
FITTED_BASELINES: tuple[str, ...] = ("random", "class_freq", "periph", "gd_rule")

#: §6a clause 2B: the non-inferiority margin and the one-sided 95% quantile.
NI_MARGIN = 0.05
NI_Z = 1.6448536269514722                  # Phi^-1(0.95)
#: 1.645 + 0.842 — the constant the prereg's own sizing table is written with
#: (``n = ((1.645+0.842)*sd/(Delta+delta))^2``), reused for n80 and for the
#: measured MDE so that every power number in the round comes off one constant.
N80_Z = 2.4866

#: §6a clause 3: a cell is judged only if it carries this many live parents
#: (the number is ``class_freq``'s within-cell n80 = 9.9, not a round number
#: chosen for looking like one) and is RANKABLE.
CELL_MIN_LIVE = 10
#: §6a clause 3 (ii): at least one of the five baselines must leave the cell's
#: within-parent permutation null at this level.  v3.1's own score is not used.
RANKABILITY_ALPHA = 0.05
#: §6a clause 3, cell-concentration: one cell owning more than this share of the
#: live parents that decide clause 2B VOIDS the verdict.
CELL_SHARE_MAX = 0.40

#: §6a clause 4: the served Platt spread, ``policy_v2``'s 0.160 to one figure.
SERVING_SPREAD_MIN = 0.15
#: §5c: REPORTED, never gated.  Kept as a named constant only so the report can
#: print the line the prereg refused to gate and say that it refused.
ECE_REPORT_LINE = 0.05

#: §6d: a clause-2B FAIL under 80% realized power is UNDECIDED, not FAIL.
POWER_FLOOR = 0.80

#: The `mde_screen` convention of §3c-(4)'s own table: a two-sided 95%
#: detection threshold on the permutation null, ``1.96 * sd(null mean)``.  It is
#: carried BESIDE the 80%-power MDE because the prereg's registered numbers
#: (f109 0.160, f113 0.142, HGD569 0.227) are in this convention -- f109's
#: 0.2010 at 80% power and 0.1584 at this one are the SAME measurement, and a
#: reader comparing the wrong one to the registered <= 0.20 bar would think the
#: qualifying cell had stopped qualifying.
MDE_SCREEN_Z = 1.959963984540054

#: Permutation reps for the rankability test and the MDE.  Fixed, not a knob.
PERM_REPS = 2000
PERM_SEED = 20260903


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path) -> str:
    """The sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def splits_fingerprint_ok(path: str | Path,
                          expected: str = SPLITS_V31_SHA256) -> bool:
    """``True`` iff ``path`` is the frozen STEP 0-a assignment."""
    p = Path(path)
    return p.is_file() and sha256_file(p) == expected


def assert_splits_registered(path: str | Path,
                             expected: str = SPLITS_V31_SHA256) -> str:
    """Return the sha256 of ``path``, or refuse.

    The gate's power arithmetic (§6a clause 2B) is stated on ONE realized
    assignment — 72 parents with >= 8 F_xy candidates, 39 live — and a run that
    re-derives the split instead of reading the frozen one can move that number
    without leaving a trace in ``metrics.json``.  So the file is hashed, not
    trusted.
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"the v3.1 gate is computed on the FROZEN cross-fit assignment and "
            f"{p} does not exist.  Emit it first:\n"
            f"  python -m lpopt.policy.train_v3 --steps "
            f"data/policy/steps_v31.parquet --holdout-cell "
            f'"{PROSPECTIVE_CELL_V31}" --xfit-k 5 --base-seed 20260903 '
            f"--device cpu --out-dir data/policy/v31_split")
    got = sha256_file(p)
    if got != expected:
        raise SystemExit(
            f"{p} hashes {got}, not the registered {expected} (freeze stamp "
            f"§S0.5).  The realized fold table, the gate pool's live count and "
            f"clause 2B's power are all stated on those bytes; a different "
            f"assignment is a different round and is refused rather than "
            f"scored.")
    return got


def load_splits(path: str | Path, *, verify: bool = True) -> pd.DataFrame:
    """Read ``splits_v31.csv`` back into the ``DataFrame[fold, xfit_fold]`` shape.

    ``emit_crossfit_splits`` writes ``child_record_id`` first for traceability;
    the two columns the fold helpers read are ``fold`` and ``xfit_fold`` and the
    row ORDER is the corpus order, which is what makes positional indexing
    valid.  The caller checks that order against the corpus.
    """
    if verify:
        assert_splits_registered(path)
    frame = pd.read_csv(path)
    for col in ("fold", "xfit_fold"):
        if col not in frame.columns:
            raise SystemExit(f"{path} has no {col!r} column; it is not a v3.1 "
                             f"split emission")
    return frame


def assert_splits_align(steps: pd.DataFrame, splits: pd.DataFrame) -> None:
    """The split file and the corpus must be the same rows in the same order."""
    if len(steps) != len(splits):
        raise SystemExit(f"{len(splits)} split rows against {len(steps)} corpus "
                         f"rows; the assignment is not for this corpus")
    if "child_record_id" in splits.columns:
        a = steps["child_record_id"].astype(str).to_numpy()
        b = splits["child_record_id"].astype(str).to_numpy()
        if not np.array_equal(a, b):
            n = int((a != b).sum())
            raise SystemExit(
                f"the split assignment is row-aligned by POSITION and {n} rows "
                f"carry a different child_record_id than the corpus does; the "
                f"corpus was re-mined or re-ordered under a frozen split")


# --------------------------------------------------------------------------- #
# the registered gain — §5a, and v3 deviation §1.6 closed from this side
# --------------------------------------------------------------------------- #
def registered_gain(steps: pd.DataFrame) -> np.ndarray:
    """``y_fxy`` — the registered gain of §2a, zero where it is unlabelled.

    ``targets_v3`` is the single definition of the three constants
    (``CYCLEN_TOL`` 5.0, ``F_R_LIMIT`` 1.55, ``c_fxy`` 0.060); it is called
    rather than reproduced so that a change to any of them is a change to one
    place and, per §2a, a new round.
    """
    y, m = targets_v3(steps)
    return np.where(m[:, 2] > 0, y[:, 2], 0.0)


def assert_registered_gain(gain: np.ndarray, steps: pd.DataFrame,
                           *, atol: float = 1e-9) -> np.ndarray:
    """Refuse a ranking statistic that was handed the RAW gain (§5a, tests H(c)).

    v3's deviation §1.6 was that the consumer metric was computed on raw
    ``-d_f_xy`` while the gate claimed the registered ``y_fxy``, and the two
    disagree in SIGN on the v3-vs-v2 comparison.  §5a's answer is that the
    metric module must be impossible to feed the wrong one, so every entry point
    here validates rather than documents:

    * the registered gain lives in ``[0, 1]`` by construction (it is clipped at
      ``c_fxy`` and divided by it) — raw gain is in EFPD-scale physical units and
      is negative for a worsening move;
    * and it must equal :func:`registered_gain` row for row, so a rescaled raw
      column that happens to land in ``[0, 1]`` is caught too.
    """
    g = np.asarray(gain, dtype=np.float64).ravel()
    if g.size != len(steps):
        raise ValueError(f"the gain column has {g.size} rows against the "
                         f"frame's {len(steps)}")
    if not np.isfinite(g).all():
        raise ValueError("a non-finite gain reached a v3.1 ranking statistic; "
                         "the registered gain y_fxy is finite everywhere "
                         "(unlabelled rows are 0)")
    if g.min() < -atol or g.max() > 1.0 + atol:
        raise ValueError(
            f"a v3.1 ranking statistic was fed a gain in "
            f"[{g.min():.4g}, {g.max():.4g}], which is not the registered "
            f"y_fxy in [0, 1] (§2a).  The raw gain -d_f_xy is NOT the gate's "
            f"gain: v3 deviation §1.6 is that the two disagree in sign on the "
            f"v3-vs-v2 comparison, and §5a registers that every ranking "
            f"statistic here is fed y_fxy.")
    want = registered_gain(steps)
    if not np.allclose(g, want, atol=1e-9, rtol=0.0):
        n = int((~np.isclose(g, want, atol=1e-9, rtol=0.0)).sum())
        raise ValueError(
            f"the gain column differs from the registered y_fxy on {n} of "
            f"{len(g)} rows.  §5a admits exactly one gain into the gate "
            f"statistics; a monotone re-scaling of the raw gain is still not it.")
    return g


# --------------------------------------------------------------------------- #
# §5d — the four baselines, refitted inside each block
# --------------------------------------------------------------------------- #
def oof_baseline_scores(steps: pd.DataFrame, splits: pd.DataFrame, *,
                        head: str = "fxy", v2: pd.DataFrame | None = None,
                        seed: int = PERM_SEED,
                        ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """The stitched out-of-fold baseline columns — **prereg delta D itself**.

    For each block ``b``:

    1. the fit fold is ``xfit_indices(splits, b)["train"]`` — every OTHER pool
       block plus the legacy ``train`` fold.  It contains no row of block ``b``,
       no ``val`` component and no row of the held-out cell;
    2. ``gd_rule``'s sign and ``class_freq``'s per-class rates are fitted on
       THAT fold by the v3 code (:func:`gd_rule_sign`,
       :func:`baseline_scores_v3`), unchanged;
    3. block ``b``'s eval rows are scored with that fit and written into the
       output column at their own positions.

    The result is a full-length column per baseline whose pool entries are all
    out of fold and whose non-pool entries are ``NaN``.  ``policy_v2`` is passed
    through unchanged in every block: it is a frozen checkpoint's blind CSV
    (§3b), so "refitting" it has no meaning and pretending otherwise would let a
    block's rows influence the baseline they are scored against.

    Returns ``(columns, per_block_fits)``; the second element is what
    ``metrics.json`` carries so a reader can see the K fits actually differed.
    """
    assert_splits_align(steps, splits)
    n = len(steps)
    cols = {k: np.full(n, np.nan) for k in BASELINES}
    fits: list[dict[str, Any]] = []
    blocks = sorted({int(b) for b in
                     splits.loc[splits["fold"] == "pool", "xfit_fold"]})
    seen = np.zeros(n, bool)
    for b in blocks:
        idx = xfit_indices(splits, b)
        tr = steps.iloc[idx["train"]]
        ev = idx["eval"]
        if not len(ev):
            continue
        overlap = np.intersect1d(idx["train"], ev)
        if overlap.size:
            raise SystemExit(
                f"block {b}'s fit fold and its eval fold share {overlap.size} "
                f"rows; the out-of-fold column would be fitted on the rows it "
                f"scores and §5d's refit means nothing")
        if seen[ev].any():
            raise SystemExit(f"block {b} re-scores rows another block already "
                             f"scored; the blocks are not a partition")
        seen[ev] = True
        gd = gd_rule_sign(tr)
        sc = baseline_scores_v3(steps.iloc[ev], tr, head, v2, gd_sign=gd,
                                seed=seed + b)
        for k in BASELINES:
            cols[k][ev] = sc[k]
        lab = tr[EVAL_LABEL_V3[head]]
        fits.append({
            "block": int(b), "n_fit_rows": int(len(idx["train"])),
            "n_eval_rows": int(len(ev)), "gd_rule_sign": float(gd),
            "fit_base_rate": (float(lab.mean()) if lab.notna().any()
                              else float("nan")),
            "n_fit_labelled": int(lab.notna().sum()),
        })
    return cols, fits


# --------------------------------------------------------------------------- #
# small statistics
# --------------------------------------------------------------------------- #
def n80(delta: float, sd: float, *, margin: float = 0.0) -> float:
    """``((1.645+0.842)*sd/(delta+margin))^2`` — the prereg's own sizing formula."""
    denom = float(delta) + float(margin)
    if not np.isfinite(denom) or denom <= 0.0 or not np.isfinite(sd):
        return float("inf")
    return float((N80_Z * float(sd) / denom) ** 2)


def normal_power(delta: float, sd: float, n: int, *, margin: float = 0.0,
                 z: float = NI_Z) -> float:
    """One-sided power at ``n`` parents for a true effect ``delta``."""
    if n <= 0 or not np.isfinite(sd) or sd <= 0.0:
        return float("nan")
    arg = (float(delta) + float(margin)) * math.sqrt(n) / float(sd) - z
    return float(0.5 * (1.0 + math.erf(arg / math.sqrt(2.0))))


def _parent_groups(parents: np.ndarray, gain: np.ndarray, *,
                   min_candidates: int, k: int) -> list[np.ndarray]:
    """The parents a listwise statistic is defined on: >= ``min_candidates``
    candidates and a non-zero ideal gain."""
    order = np.argsort(parents, kind="mergesort")
    p_sorted = parents[order]
    bounds = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    disc = 1.0 / np.log2(np.arange(k) + 2.0)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[a:b]
        if len(idx) < min_candidates:
            continue
        g = np.clip(gain[idx], 0.0, None)
        if float((np.sort(g)[::-1][:k] * disc[:min(k, len(g))]).sum()) <= 0.0:
            continue
        out.append(idx)
    return out


def _ndcg_once(scores: np.ndarray, gain: np.ndarray, groups: list[np.ndarray],
               *, k: int, rng: np.random.Generator) -> np.ndarray:
    """One tie-broken NDCG@k draw per group — the permutation loop's inner step."""
    disc = 1.0 / np.log2(np.arange(k) + 2.0)
    out = np.empty(len(groups))
    for j, idx in enumerate(groups):
        g = np.clip(gain[idx], 0.0, None)
        ideal = float((np.sort(g)[::-1][:k] * disc[:min(k, len(g))]).sum())
        take = np.lexsort((rng.random(len(idx)), -scores[idx]))[:k]
        out[j] = float((g[take] * disc[:len(take)]).sum()) / ideal
    return out


def within_parent_permutation(scores: np.ndarray, gain: np.ndarray,
                              parents: np.ndarray, *, k: int = NDCG_K,
                              min_candidates: int = REGRET_MIN_CANDIDATES,
                              reps: int = PERM_REPS, seed: int = PERM_SEED,
                              ) -> dict[str, float]:
    """The §6a clause 3-(ii) null: shuffle each parent's scores among ITS OWN rows.

    Blocking the permutation inside the parent is what makes the null "this cell
    carries no within-parent order", which is the question clause 3 asks, rather
    than "this cell's parents differ", which it does not.

    Returns the observed mean NDCG@k, the null mean and sd, a one-sided p-value,
    and the MEASURED MDE — ``2.4866 * sd(null mean)``, the same 1.645 + 0.842
    the prereg's sizing table uses.  §3c-(4)'s closed form ``0.55/sqrt(n_live)``
    is the approximation this replaces, and both are reported.
    """
    groups = _parent_groups(parents, gain, min_candidates=min_candidates, k=k)
    if not groups:
        return {"n_parents": 0, "observed": float("nan"), "null_mean": float("nan"),
                "null_sd": float("nan"), "p_value": float("nan"),
                "mde": float("nan"), "mde_screen_z196": float("nan"),
                "mde_closed_form": float("nan"),
                "reps": int(reps)}
    rng = np.random.default_rng(seed)
    obs = float(_ndcg_once(scores, gain, groups, k=k, rng=rng).mean())
    null = np.empty(reps)
    shuffled = np.array(scores, dtype=np.float64, copy=True)
    for r in range(reps):
        for idx in groups:
            shuffled[idx] = rng.permutation(scores[idx])
        null[r] = float(_ndcg_once(shuffled, gain, groups, k=k, rng=rng).mean())
    sd = float(null.std(ddof=1))
    n = len(groups)
    return {"n_parents": int(n), "observed": obs, "null_mean": float(null.mean()),
            "null_sd": sd, "p_value": float((1.0 + (null >= obs).sum())
                                            / (1.0 + reps)),
            "mde": float(N80_Z * sd),
            "mde_screen_z196": float(MDE_SCREEN_Z * sd),
            "mde_closed_form": float(0.55 / math.sqrt(n)) if n else float("nan"),
            "reps": int(reps)}


def cell_rankability(scores: dict[str, np.ndarray], gain: np.ndarray,
                     parents: np.ndarray, *,
                     reps: int = PERM_REPS, seed: int = PERM_SEED,
                     alpha: float = RANKABILITY_ALPHA) -> dict[str, Any]:
    """§6a clause 3-(ii) for ONE cell, over the five baselines.

    **v3.1's own score is not read here.**  The cell either carries a signal a
    pre-existing scorer can find, or it is UNDECIDABLE, and deciding that with
    the model under test would be the f113 error (results §3.4) run backwards.
    """
    per: dict[str, dict[str, float]] = {}
    for i, name in enumerate(BASELINES):
        s = np.asarray(scores[name], float)
        if not np.isfinite(s).any():
            per[name] = {"p_value": float("nan"), "observed": float("nan")}
            continue
        finite = s[np.isfinite(s)]
        s = np.where(np.isfinite(s), s, float(finite.min()) - 1.0)
        per[name] = within_parent_permutation(s, gain, parents, reps=reps,
                                              seed=seed + 7919 * i)
    ps = [v["p_value"] for v in per.values() if np.isfinite(v["p_value"])]
    return {"per_baseline": per,
            "min_p": float(min(ps)) if ps else float("nan"),
            "rankable": bool(ps and min(ps) <= alpha)}


# --------------------------------------------------------------------------- #
# §6a — the clauses
# --------------------------------------------------------------------------- #
def clause_1_parent_blocked_auc(scores: dict[str, np.ndarray], y: np.ndarray,
                                parents: np.ndarray, *, gate_auc: float,
                                ci_lo: float) -> dict[str, Any]:
    """parent-blocked AUC >= ``gate_auc`` with a paired-parent CI lower bound
    above ``ci_lo``."""
    names = list(scores)
    pb, npairs = parent_blocked_auc(scores["policy"], y, parents)
    out: dict[str, Any] = {
        "policy": float(pb), "n_pairs": int(npairs),
        "per_scorer": {k: float(parent_blocked_auc(scores[k], y, parents)[0])
                       for k in names},
    }
    per = _per_parent_blocked(scores, y, parents, names)
    if per is not None:
        summary, boots = _paired_parent_bootstrap(per)
        out["ci"] = summary["policy"]
        out["n_mixed_parents"] = int(len(per["policy"]))
        out["delta"] = {b: _delta(boots["policy"] - boots[b],
                                  float(np.nanmean(per["policy"] - per[b])))
                        for b in BASELINES if b in per}
    lo = float(out.get("ci", {}).get("lo", float("nan")))
    out["threshold"] = {"auc": gate_auc, "ci_lo": ci_lo}
    out["PASS"] = bool(np.isfinite(pb) and pb >= gate_auc
                       and np.isfinite(lo) and lo > ci_lo)
    return out


def _ndcg_table(scores: dict[str, np.ndarray], gain: np.ndarray,
                parents: np.ndarray, *, k: int = NDCG_K,
                min_candidates: int = REGRET_MIN_CANDIDATES,
                ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-parent NDCG@k for every scorer, on one common parent set."""
    per: dict[str, np.ndarray] = {}
    keys: np.ndarray | None = None
    for name, s in scores.items():
        s = np.asarray(s, float)
        if not np.isfinite(s).all():
            finite = s[np.isfinite(s)]
            floor = (float(finite.min()) - 1.0) if finite.size else 0.0
            s = np.where(np.isfinite(s), s, floor)
        vals, kk = ndcg_at_k(s, gain, parents, k=k,
                             min_candidates=min_candidates)
        per[name] = vals
        keys = kk if keys is None else keys
    return per, (keys if keys is not None else np.array([], dtype=object))


def clause_2a_ndcg(per_parent: dict[str, np.ndarray]) -> dict[str, Any]:
    """§6a clause 2A — NDCG@4-of-8 above all four FITTED baselines, each paired
    CI excluding 0.  A regression clause: §6a records that it already passes."""
    summary, boots = _paired_parent_bootstrap(per_parent)
    out: dict[str, Any] = {"ndcg": summary, "delta": {}, "n80": {},
                           "n_informative": {}}
    for b in FITTED_BASELINES:
        if b not in per_parent:
            continue
        d = per_parent["policy"] - per_parent[b]
        out["delta"][b] = _delta(boots["policy"] - boots[b], float(np.nanmean(d)))
        sd = float(np.nanstd(d, ddof=1)) if len(d) > 1 else float("nan")
        out["delta"][b]["sd"] = sd
        out["n80"][b] = n80(float(np.nanmean(d)), sd)
        out["n_informative"][b] = int(np.count_nonzero(
            np.isfinite(d) & (np.abs(d) > 0.0)))
    out["n_parents"] = int(len(per_parent["policy"]))
    out["PASS"] = bool(out["delta"] and all(
        out["delta"][b]["beats"] for b in FITTED_BASELINES if b in out["delta"]))
    return out


def clause_2b_noninferiority(per_parent: dict[str, np.ndarray], *,
                             margin: float = NI_MARGIN, z: float = NI_Z,
                             reps: int = 4000, seed: int = 20260817,
                             ) -> dict[str, Any]:
    """§6a clause 2B — non-inferiority to ``policy_v2``, one-sided 95%.

    The DECISION is the normal-approximation lower bound the prereg's own sizing
    table is written in (``Delta - z*sd/sqrt(n) > -margin``); the paired
    bootstrap's 5th percentile is reported beside it so the two can be seen to
    agree or not.  Realized power is computed at the observed effect AND at a
    true effect of zero, because §6d's UNDECIDED rule turns on it: a clause-2B
    FAIL under 80% power is recorded as UNDECIDED, not as a FAIL.
    """
    if "policy_v2" not in per_parent:
        return {"available": False, "PASS": False, "verdict": "UNDECIDED",
                "reason": "no policy_v2 baseline column"}
    d = per_parent["policy"] - per_parent["policy_v2"]
    d = d[np.isfinite(d)]
    n = int(len(d))
    if n < 2:
        return {"available": False, "PASS": False, "verdict": "UNDECIDED",
                "reason": f"{n} paired parents"}
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    half = z * sd / math.sqrt(n)
    lo = mean - half
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(reps, n))
    boot = d[draws].mean(axis=1)
    power_obs = normal_power(mean, sd, n, margin=margin, z=z)
    power_null = normal_power(0.0, sd, n, margin=margin, z=z)
    passed = bool(lo > -margin)
    return {
        "available": True, "delta": mean, "sd": sd, "n_parents": n,
        "n_informative": int(np.count_nonzero(np.abs(d) > 0.0)),
        "lo_one_sided_95": float(lo),
        "lo_bootstrap_5pct": float(np.percentile(boot, 5.0)),
        "margin": float(margin),
        "n_ci": n80(mean, sd, margin=margin) * (z / N80_Z) ** 2,
        "n80": n80(mean, sd, margin=margin),
        "n80_at_zero": n80(0.0, sd, margin=margin),
        "power_at_observed": power_obs, "power_at_zero": power_null,
        "PASS": passed,
        # §6d: a FAIL that the round could not have detected is not a FAIL.
        "verdict": ("PASS" if passed else
                    ("FAIL" if (np.isfinite(power_obs)
                                and power_obs >= POWER_FLOOR) else "UNDECIDED")),
    }


def clause_3_within_cell(frame: pd.DataFrame, scores: dict[str, np.ndarray],
                         gain: np.ndarray, parents: np.ndarray, *,
                         margin: float = NI_MARGIN, min_live: int = CELL_MIN_LIVE,
                         reps: int = PERM_REPS, seed: int = PERM_SEED,
                         ) -> dict[str, Any]:
    """§6a clause 3 — the within-cell ranking clause r2's adjudication demanded.

    For each cell: ``class_freq`` and ``gd_rule`` must be beaten with a paired CI
    excluding 0, and ``policy_v2`` must not be lost to by more than ``margin``.
    A cell is JUDGED only if it carries ``min_live`` live parents and is RANKABLE
    (clause 3-(ii)); otherwise it is UNDECIDABLE and — the half of this that had
    to be written before the numbers existed — **it does not count as a FAIL.**
    """
    cells = frame["cell"].astype(str).to_numpy()
    out: dict[str, Any] = {"cells": {}, "min_live": int(min_live),
                           "margin": float(margin)}
    for cell in sorted(set(cells)):
        m = cells == cell
        g, p = gain[m], parents[m]
        sc = {k: np.asarray(v, float)[m] for k, v in scores.items()}
        groups = _parent_groups(p, g, min_candidates=REGRET_MIN_CANDIDATES,
                                k=NDCG_K)
        n_live = len(groups)
        entry: dict[str, Any] = {"n_rows": int(m.sum()), "n_live": int(n_live)}
        if n_live < min_live:
            entry["verdict"] = "UNDECIDABLE"
            entry["reason"] = f"{n_live} live parents < {min_live} (clause 3-i)"
            out["cells"][cell] = entry
            continue
        rank = cell_rankability({k: v for k, v in sc.items() if k in BASELINES},
                                g, p, reps=reps, seed=seed)
        entry["rankability"] = rank
        if not rank["rankable"]:
            entry["verdict"] = "UNDECIDABLE"
            entry["reason"] = (f"no baseline leaves the within-parent "
                               f"permutation null at p <= {RANKABILITY_ALPHA} "
                               f"(min p {rank['min_p']:.3f}); clause 3-ii")
            out["cells"][cell] = entry
            continue
        per, _ = _ndcg_table(sc, g, p)
        summary, boots = _paired_parent_bootstrap(per)
        entry["ndcg"] = summary
        entry["delta"] = {}
        for b in ("class_freq", "gd_rule"):
            if b in per:
                entry["delta"][b] = _delta(
                    boots["policy"] - boots[b],
                    float(np.nanmean(per["policy"] - per[b])))
        beats = all(entry["delta"][b]["beats"] for b in entry["delta"])
        ni = clause_2b_noninferiority(per, margin=margin)
        entry["vs_policy_v2"] = ni
        not_worse = bool(ni.get("PASS")) if ni.get("available") else True
        entry["verdict"] = "PASS" if (beats and not_worse) else "FAIL"
        out["cells"][cell] = entry
    judged = {c: e for c, e in out["cells"].items()
              if e["verdict"] in ("PASS", "FAIL")}
    out["n_eligible"] = len(judged)
    out["undecidable"] = sorted(c for c, e in out["cells"].items()
                                if e["verdict"] == "UNDECIDABLE")
    out["PASS"] = bool(judged) and all(e["verdict"] == "PASS"
                                       for e in judged.values())
    return out


def cell_concentration(frame: pd.DataFrame, gain: np.ndarray,
                       parents: np.ndarray, *, max_share: float = CELL_SHARE_MAX
                       ) -> dict[str, Any]:
    """§6a clause 3's cell-bias clause: one cell owning > 40% of the live parents
    that decide clause 2B VOIDS the verdict."""
    groups = _parent_groups(parents, gain, min_candidates=REGRET_MIN_CANDIDATES,
                            k=NDCG_K)
    cells = frame["cell"].astype(str).to_numpy()
    counts = pd.Series([cells[idx[0]] for idx in groups]).value_counts()
    total = int(counts.sum())
    share = (counts / total).to_dict() if total else {}
    top = max(share.values()) if share else float("nan")
    return {"n_live": total, "share": {k: float(v) for k, v in share.items()},
            "max_share": float(top), "max_cell": (counts.idxmax() if total
                                                  else None),
            "max_share_allowed": float(max_share),
            "VOID": bool(total and top > max_share)}


def clause_4_serving_scale(logits: np.ndarray, y: np.ndarray,
                           splits: pd.DataFrame, pool: np.ndarray, *,
                           spread_min: float = SERVING_SPREAD_MIN
                           ) -> dict[str, Any]:
    """§6a clause 4 — the Platt-served p90-p10 on the gate pool.

    The map is fitted by :func:`~.v3.fit_platt_v31`, which takes its fold from
    :func:`~.v3.calib_index` and not from this caller, so ``val`` and the
    held-out cell cannot reach it however this function is called (§10-1).
    """
    platt = fit_platt_v31(logits, y, splits)
    served = platt_serve(np.asarray(logits, float), a=platt["a"], b=platt["b"])
    gp = served[pool]
    raw = 1.0 / (1.0 + np.exp(-np.asarray(logits, float)[pool]))
    q = lambda v: float(np.percentile(v, 90) - np.percentile(v, 10))  # noqa: E731
    return {"platt": platt, "spread_served": q(gp), "spread_raw": q(raw),
            "spread_logit": q(np.asarray(logits, float)[pool]),
            "spread_min": float(spread_min),
            "PASS": bool(q(gp) >= spread_min)}


def calibration_report(probs: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """ECE / Brier — **reported, not gated** (§5c).

    §5c registers the reason before the fact: the two measured configurations
    (0.0521 raw, 0.0858 val-Platt) are both already above 0.05, so gating the
    line would be registering a FAIL that is known in advance.  The flag is
    carried so a reader can see the number against the line, and ``gated`` says
    plainly that it decided nothing.
    """
    out = calibration(np.asarray(probs, float), np.asarray(y, float))
    out["report_line"] = ECE_REPORT_LINE
    out["ece_le_line"] = bool(out["ece"] <= ECE_REPORT_LINE)
    out["gated"] = False
    out["why_not_gated"] = (
        "§5c: both measured configurations (0.0521 raw, 0.0858 val-Platt) are "
        "already above 0.05, so gating ECE would register a known FAIL; clause "
        "4 gates the served SPREAD instead.")
    return out


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def gate_report_v31(steps: pd.DataFrame, splits: pd.DataFrame,
                    logits: np.ndarray, *, v2: pd.DataFrame | None = None,
                    gain: np.ndarray | None = None, gate_auc: float = 0.65,
                    gate_auc_ci_lo: float = 0.50, transfer_auc: float = 0.60,
                    perm_reps: int = PERM_REPS, seed: int = PERM_SEED,
                    ) -> dict[str, Any]:
    """Every §6 number, out of fold, on the frozen assignment.

    ``logits`` is the ensemble's ``fxy`` LOGIT for every corpus row — out of fold
    on the pool (each row scored by the member that never trained on it) and
    in-sample nowhere that is judged.  Probabilities are not accepted: §5c's
    whole point is that the sigmoid is a serving artefact and the scale question
    is asked on the logit.
    """
    assert_splits_align(steps, splits)
    z = np.asarray(logits, float).ravel()
    if z.size != len(steps):
        raise ValueError(f"{z.size} logits for {len(steps)} corpus rows")
    g = registered_gain(steps) if gain is None else assert_registered_gain(
        gain, steps)

    fold = splits["fold"].to_numpy()
    pool = np.flatnonzero(fold == "pool")
    cell_fold = np.flatnonzero(fold == "prospective_cell")
    parents = steps["parent_record_id"].astype(str).to_numpy()
    label = steps[EVAL_LABEL_V3["fxy"]]
    # ``label`` is a masked BOOLEAN column, so ``fillna(0)`` is a dtype error
    # (pandas 3.0 rejects the int fill on a boolean array).  Cast straight to
    # float with the missing rows sent to 0.0 -- unlabelled rows are never
    # judged; every consumer below indexes through ``label.notna()``.
    y_all = label.to_numpy(dtype=float, na_value=0.0)

    base_cols, fits = oof_baseline_scores(steps, splits, head="fxy", v2=v2,
                                          seed=seed)
    report: dict[str, Any] = {
        "prereg": "policy_v31_prereg_20260831_DRAFT.md",
        "delta_D": "lpopt.policy.metrics_v31",
        "n_blocks": len(fits), "block_fits": fits,
        "pool": {"n_rows": int(len(pool)),
                 "n_fxy": int(label.notna().to_numpy()[pool].sum()),
                 "n_live_floor": MIN_POOL_LIVE_V31},
    }

    # ---- the judged rows: pool, F_xy-labelled ------------------------------ #
    lab = label.notna().to_numpy()
    idx = pool[lab[pool]]
    scores = {"policy": z[idx], **{k: base_cols[k][idx] for k in BASELINES}}
    for k, v in scores.items():
        if np.isnan(v).any():
            finite = v[np.isfinite(v)]
            scores[k] = np.where(np.isfinite(v),
                                 v, (float(finite.min()) - 1.0
                                     if finite.size else 0.0))
    y = y_all[idx]
    par = parents[idx]
    gg = g[idx]

    report["clause_1"] = clause_1_parent_blocked_auc(
        scores, y, par, gate_auc=gate_auc, ci_lo=gate_auc_ci_lo)
    per_parent, _keys = _ndcg_table(scores, gg, par)
    report["clause_2A"] = clause_2a_ndcg(per_parent)
    report["clause_2B"] = clause_2b_noninferiority(per_parent)
    report["clause_3"] = clause_3_within_cell(
        steps.iloc[idx], scores, gg, par, reps=perm_reps, seed=seed)
    report["cell_concentration"] = cell_concentration(steps.iloc[idx], gg, par)
    report["clause_4"] = clause_4_serving_scale(z, y_all, splits, idx)
    served = platt_serve(z, a=report["clause_4"]["platt"]["a"],
                         b=report["clause_4"]["platt"]["b"])
    report["calibration"] = calibration_report(served[idx], y)
    report["mde"] = within_parent_permutation(scores["policy"], gg, par,
                                              reps=perm_reps, seed=seed)

    report["PASS"] = bool(
        report["clause_1"]["PASS"] and report["clause_2A"]["PASS"]
        and report["clause_2B"]["PASS"] and report["clause_3"]["PASS"]
        and report["clause_4"]["PASS"]
        and not report["cell_concentration"]["VOID"])
    report["note"] = ("clause 5 (fr/flat regression) is STRUCTURAL and is "
                      "asserted at checkpoint level by "
                      "train_v3.assert_stage2_init_is_stage1; it is not a fold "
                      "statistic and is not computed here.")

    # ---- the transfer bar (§6b), opened once after the gate exists --------- #
    if len(cell_fold):
        tidx = cell_fold[lab[cell_fold]]
        if len(tidx):
            # The held-out cell is in NO block, so there is nothing to stitch:
            # every block's fit is out of fold for it.  Block 0's fit is used
            # and the choice is recorded rather than averaged, because averaging
            # K baseline fits is not any of the K baselines.
            b0 = xfit_indices(splits, 0)
            tsc = {"policy": z[tidx],
                   **baseline_scores_v3(steps.iloc[tidx],
                                        steps.iloc[b0["train"]], "fxy", v2,
                                        gd_sign=gd_rule_sign(
                                            steps.iloc[b0["train"]]),
                                        seed=seed)}
            for k, v in tsc.items():
                v = np.asarray(v, float)
                if np.isnan(v).any():
                    finite = v[np.isfinite(v)]
                    v = np.where(np.isfinite(v), v,
                                 (float(finite.min()) - 1.0) if finite.size
                                 else 0.0)
                tsc[k] = v
            tper, _ = _ndcg_table(tsc, g[tidx], parents[tidx])
            tpb = float(parent_blocked_auc(tsc["policy"], y_all[tidx],
                                           parents[tidx])[0])
            tsum, tboots = _paired_parent_bootstrap(tper)
            tdelta = {b: _delta(tboots["policy"] - tboots[b],
                                float(np.nanmean(tper["policy"] - tper[b])))
                      for b in FITTED_BASELINES if b in tper}
            report["transfer_bar"] = {
                "cell": PROSPECTIVE_CELL_V31, "n_rows": int(len(tidx)),
                "baseline_fit_block": 0,
                "pb_auc": tpb, "threshold_auc": transfer_auc,
                "ndcg": tsum, "delta": tdelta,
                # §6b: policy_v2 is REPORTED here and explicitly not gated —
                # 16 live parents against the 113 a CI exclusion needs.
                "vs_policy_v2_reported_only": clause_2b_noninferiority(tper),
                "mde": within_parent_permutation(tsc["policy"], g[tidx],
                                                 parents[tidx], reps=perm_reps,
                                                 seed=seed),
                "PASS": bool(np.isfinite(tpb) and tpb >= transfer_auc
                             and tdelta
                             and all(d["beats"] for d in tdelta.values())),
            }
    return report


def render_gate(report: dict[str, Any]) -> str:
    """A terse text rendering; the report JSON stays the authority."""
    L: list[str] = []
    c1, c2a, c2b = report["clause_1"], report["clause_2A"], report["clause_2B"]
    L.append(f"clause 1  pb-AUC {c1['policy']:.4f} "
             f"[{c1.get('ci', {}).get('lo', float('nan')):.4f}, "
             f"{c1.get('ci', {}).get('hi', float('nan')):.4f}] "
             f"mixed parents {c1.get('n_mixed_parents')} -> "
             f"{'PASS' if c1['PASS'] else 'FAIL'}")
    for b, d in c2a["delta"].items():
        L.append(f"clause 2A vs {b:<11s} d={d['mean']:+.4f} "
                 f"[{d['lo']:+.4f}, {d['hi']:+.4f}] n80={c2a['n80'][b]:.1f} "
                 f"informative={c2a['n_informative'][b]}")
    if c2b.get("available"):
        L.append(f"clause 2B vs policy_v2  d={c2b['delta']:+.4f} sd={c2b['sd']:.4f} "
                 f"lo95={c2b['lo_one_sided_95']:+.4f} (margin -{c2b['margin']}) "
                 f"n={c2b['n_parents']} power@obs={c2b['power_at_observed']:.2f} "
                 f"-> {c2b['verdict']}")
    for cell, e in report["clause_3"]["cells"].items():
        L.append(f"clause 3  {cell:<22s} live={e['n_live']:>3d} {e['verdict']}"
                 + (f"  ({e['reason']})" if "reason" in e else ""))
    c4 = report["clause_4"]
    L.append(f"clause 4  served p90-p10 {c4['spread_served']:.4f} "
             f"(raw {c4['spread_raw']:.4f}, logit {c4['spread_logit']:.3f}) -> "
             f"{'PASS' if c4['PASS'] else 'FAIL'}")
    cal = report["calibration"]
    L.append(f"report    ECE {cal['ece']:.4f} Brier {cal['brier']:.4f} "
             f"(NOT gated, §5c)")
    L.append(f"GATE = {'PASS' if report['PASS'] else 'FAIL'}")
    if "transfer_bar" in report:
        t = report["transfer_bar"]
        L.append(f"transfer  {t['cell']} pb-AUC {t['pb_auc']:.4f} -> "
                 f"{'PASS' if t['PASS'] else 'FAIL'}")
    return "\n".join(L)


__all__ = [
    "SPLITS_V31_SHA256", "STEPS_V31_SHA256", "FITTED_BASELINES", "NI_MARGIN",
    "CELL_MIN_LIVE", "CELL_SHARE_MAX", "assert_registered_gain",
    "assert_splits_align", "assert_splits_registered", "calibration_report",
    "cell_concentration", "cell_rankability", "clause_1_parent_blocked_auc",
    "clause_2a_ndcg", "clause_2b_noninferiority", "clause_3_within_cell",
    "clause_4_serving_scale", "gate_report_v31", "load_splits", "n80",
    "normal_power", "oof_baseline_scores", "registered_gain", "render_gate",
    "MDE_SCREEN_Z", "N80_Z", "sha256_file", "splits_fingerprint_ok",
    "within_parent_permutation",
]
