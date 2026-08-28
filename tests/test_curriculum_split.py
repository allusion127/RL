"""Curriculum retrain split policy + honest no-regression gate (forensic fix).

The forensic audit proved two coupled defects the plain ``make_splits`` retrain
had:

* the S1 ancestry-closure union-find, rebuilt from scratch every retrain, could
  sweep an ENTIRE non-feed-121 curriculum band into val (a seed coin-toss) and
  collapse the previously-learned cells; and
* ``_gate_no_regression`` scored on ``head`` rows that were 100% in the
  champion's train set (in-sample) but out-of-sample for the candidate — a
  phantom-drop contamination.

``make_curriculum_split`` (this module) fixes the split; ``score_no_regression_cell``
fixes the gate.  These tests exercise the exact failure mode and the invariants
the design requires (deterministic, per-cell holdout >= 70% train + non-empty,
fold membership invariant to adding a new campaign, non-121 band never ejected,
both champions scored on identical held-out ids).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pandas as pd
import pytest

from lpopt.data.schema import compute_record_id, pack_pattern
from lpopt.model.splits import SplitManifest, make_curriculum_split
from lpopt.search.genome import fresh_units_from_feed, random_genome


# --------------------------------------------------------------------------- #
# synthetic store builder (valid packed patterns, deterministic truth)
# --------------------------------------------------------------------------- #
def _truth(canon: str) -> tuple[float, float]:
    """Deterministic (cyclen, f_r) truth from a pattern's canonical string."""
    h = int(hashlib.sha1(canon.encode()).hexdigest(), 16)
    return 600.0 + (h % 1000) / 12.5, 1.4 + ((h // 1000) % 1000) / 2500.0


def _rows(dataset, campaign, feed, pair, n, *, converged=True, seed=0,
          parent=None, library="ga80"):
    out = []
    rng = random.Random(seed)
    nf = fresh_units_from_feed(int(feed))
    made = 0
    tries = 0
    while made < n and tries < n * 40:
        tries += 1
        g = random_genome(rng, pair, nf, max_shuffle_depth=2,
                          allow_single_cycle_discharge=(nf > 30))
        canon = pack_pattern(g.to_pattern())
        rid = compute_record_id(canon, library, pair)
        cyc, fr = _truth(canon)
        out.append({
            "record_id": rid, "dataset": dataset, "campaign": campaign,
            "parent_record_id": parent, "case_pair": pair, "feed": int(feed),
            "library_id": library, "pattern": canon,
            "cyclen": cyc if converged else np.nan,
            "f_r": fr if converged else np.nan,
            "converged": bool(converged),
        })
        made += 1
    return out


def _store(*, extra_cell=False) -> pd.DataFrame:
    """A synthetic store: Dataset-A/B legacy (feed 117 + 121), a P0 pathfinder
    (mixed feeds), and 2-3 curriculum cells at their own non-121 feeds."""
    rows: list[dict] = []
    # legacy A (all feed 121, two campaigns)
    rows += _rows("A", "A::cache", 121, "K1_K2", 60, seed=1)
    rows += _rows("A", "A::cache.stale", 121, "K1_K2", 40, seed=2)
    # legacy B: feed 117 (the protected eval band) + feed 121
    rows += _rows("B", "FEASIBLE_PACKAGE", 117, "K5_K6", 25, seed=3)
    rows += _rows("B", "FEASIBLE_PACKAGE", 121, "K5_K6", 25, seed=4)
    # legacy P0 pathfinder (mixed feeds, NOT a curriculum cell -> train)
    rows += _rows("P", "P0_pathfinder", 117, "J1_J2", 15, seed=5)
    rows += _rows("P", "P0_pathfinder", 121, "J1_J2", 15, seed=6)
    # curriculum cells (dataset P, campaign == cell id, own feed)
    rows += _rows("P", "5.25-5.5_f117", 117, "L1_L2", 50, seed=7)
    rows += _rows("P", "5.25-5.5_f117", 117, "L1_L2", 10, converged=False, seed=17)
    rows += _rows("P", "5.25-5.5_f109", 109, "L3_L4", 50, seed=8)
    if extra_cell:
        # a NEW campaign added later (the forensic failure trigger)
        rows += _rows("P", "5-5.25_f117", 117, "N1_N2", 80, seed=9)
    return pd.DataFrame(rows)


CELLS_BASE = ["5.25-5.5_f117", "5.25-5.5_f109"]
CELLS_PLUS = CELLS_BASE + ["5-5.25_f117"]


# --------------------------------------------------------------------------- #
# split invariants
# --------------------------------------------------------------------------- #
def test_curriculum_holdout_70_30_and_nonempty() -> None:
    df = _store()
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    g = m.groups
    for cell in CELLS_BASE:
        n_conv = g["curriculum_conv_counts"][cell]
        val = g["curriculum_val_by_cell"][cell]
        train_conv = g["curriculum_train_conv_counts"][cell]
        assert val, f"{cell} eval holdout must be non-empty"
        assert train_conv >= 0.70 * n_conv          # >= 70% converged in train
        assert len(val) <= 0.30 * n_conv + 1         # ~20% held out
    # partition sanity
    assert set(m.train_ids).isdisjoint(set(m.val_ids))
    assert set(m.train_ids) | set(m.val_ids) == set(df["record_id"].astype(str))


def test_non_121_band_never_ejected() -> None:
    """No legacy row with feed != 121 may land in val (a whole non-121 band can
    never be swept into the holdout) — the S1-at-seed-0 failure the audit found."""
    df = _store()
    valset = set(make_curriculum_split(df, cells=CELLS_BASE, seed=0).val_ids)
    val = df[df["record_id"].astype(str).isin(valset)]
    camp = val["campaign"].astype(str)
    legacy_val = val[~((val["dataset"] == "P") & camp.isin(CELLS_BASE))]
    assert (legacy_val["feed"].astype(int) == 121).all(), \
        "a non-121 legacy row was ejected into val"
    # and every feed-117 Dataset-B row is in train
    b117 = df[(df["dataset"] == "B") & (df["feed"] == 117)]["record_id"].astype(str)
    assert set(b117).isdisjoint(valset)


def test_non_121_band_never_ejected_across_all_seeds() -> None:
    df = _store()
    for seed in range(8):
        valset = set(make_curriculum_split(df, cells=CELLS_BASE, seed=seed).val_ids)
        val = df[df["record_id"].astype(str).isin(valset)]
        camp = val["campaign"].astype(str)
        legacy_val = val[~((val["dataset"] == "P") & camp.isin(CELLS_BASE))]
        assert (legacy_val["feed"].astype(int) == 121).all(), \
            f"seed {seed} ejected a non-121 legacy row"


def test_legacy_non_121_p0_rows_go_to_train() -> None:
    df = _store()
    valset = set(make_curriculum_split(df, cells=CELLS_BASE, seed=0).val_ids)
    p0_non121 = df[(df["campaign"] == "P0_pathfinder") & (df["feed"] != 121)]
    assert set(p0_non121["record_id"].astype(str)).isdisjoint(valset)


def test_fold_membership_invariant_to_new_campaign() -> None:
    """THE forensic failure mode: adding a brand-new campaign (cell) must NOT
    change the fold membership of the existing cells' rows."""
    df_base = _store(extra_cell=False)
    df_plus = _store(extra_cell=True)
    m_base = make_curriculum_split(df_base, cells=CELLS_BASE, seed=0)
    m_plus = make_curriculum_split(df_plus, cells=CELLS_PLUS, seed=0)
    for cell in CELLS_BASE:
        assert (m_base.groups["curriculum_val_by_cell"][cell]
                == m_plus.groups["curriculum_val_by_cell"][cell]), \
            f"{cell} holdout changed when a new campaign was added"
    # the new cell also gets its own non-empty holdout
    assert m_plus.groups["curriculum_val_by_cell"]["5-5.25_f117"]


def test_deterministic() -> None:
    df = _store()
    a = make_curriculum_split(df, cells=CELLS_BASE, seed=3)
    b = make_curriculum_split(df, cells=CELLS_BASE, seed=3)
    assert a.to_dict() == b.to_dict()


def test_blind_probe_ids_pinned_into_val() -> None:
    """Blind-probe record_ids that exist as store rows are ALWAYS pinned into the
    cell's val, regardless of their stable-hash rank."""
    df = _store()
    cell = "5.25-5.5_f117"
    conv = df[(df["campaign"] == cell) & (df["converged"])]["record_id"].astype(str).tolist()
    # pick a row that the plain 20% hash holdout would NOT pick, and pin it
    m0 = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    not_held = [r for r in conv if r not in set(m0.groups["curriculum_val_by_cell"][cell])]
    assert not_held
    pin = not_held[0]
    m1 = make_curriculum_split(df, cells=CELLS_BASE, seed=0,
                               blind_probe_ids_by_cell={cell: [pin]})
    assert pin in m1.groups["curriculum_val_by_cell"][cell]
    assert pin in m1.groups["blind_probe_pins"][cell]
    # holdout still ~20% (pin displaces a hash pick, not adds on top)
    assert len(m1.groups["curriculum_val_by_cell"][cell]) \
        == len(m0.groups["curriculum_val_by_cell"][cell])


def test_manifest_json_roundtrip(tmp_path) -> None:
    df = _store()
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    p = tmp_path / "S1.json"
    m.to_json(p)
    reloaded = SplitManifest.from_json(p)
    assert reloaded.to_dict() == m.to_dict()
    assert reloaded.groups["curriculum_val_by_cell"] == m.groups["curriculum_val_by_cell"]
    assert reloaded.record_ids("val") == m.val_ids


def test_cell_cap_recorded_in_manifest_groups(tmp_path) -> None:
    """The per-curriculum sampling-weight cap is threaded to the trainer through
    the manifest: ``groups['curriculum_cell_cap']`` (the cap) + ``groups['cells']``
    (the curriculum campaign ids).  Default (no cap) records ``None`` so legacy
    retrains are untouched, and the value round-trips through JSON."""
    df = _store()
    m_none = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    assert m_none.groups["curriculum_cell_cap"] is None

    m_cap = make_curriculum_split(df, cells=CELLS_BASE, seed=0, cell_cap=16.0)
    assert m_cap.groups["curriculum_cell_cap"] == pytest.approx(16.0)
    # the trainer identifies the curriculum rows from groups['cells']
    assert set(m_cap.groups["cells"]) == set(CELLS_BASE)
    p = tmp_path / "S1_cap.json"
    m_cap.to_json(p)
    reloaded = SplitManifest.from_json(p)
    assert reloaded.groups["curriculum_cell_cap"] == pytest.approx(16.0)


def test_compute_cell_weights_reads_manifest_curriculum_cap(tmp_path) -> None:
    """End-to-end store-side threading: the cap recorded by the split is exactly
    what the trainer feeds ``compute_cell_weights`` (dataset==P curriculum rows get
    the raised cap; the legacy corpus keeps the global cap)."""
    from lpopt.model.dataset_torch import compute_cell_weights
    df = _store().copy()
    # give the frame the columns compute_cell_weights bins on
    df["e_core"] = 5.35
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0, cell_cap=16.0)
    curr_campaigns = m.groups["cells"]
    curr_cap = m.groups["curriculum_cell_cap"]
    _w, summary = compute_cell_weights(
        df, cap=8.0, curriculum_campaigns=curr_campaigns, curriculum_cap=curr_cap)
    # every curriculum-cell converged/non-converged P row is counted as curriculum
    n_curr_rows = int(((df["dataset"] == "P")
                       & df["campaign"].astype(str).isin(set(curr_campaigns))).sum())
    assert summary["curriculum_cap"] == pytest.approx(16.0)
    assert summary["n_curriculum_rows"] == n_curr_rows > 0


def test_lone_converged_row_kept_in_train() -> None:
    """A degenerate 1-converged-row cell keeps its irreplaceable in-band row in
    train (no assert blow-up, empty store holdout is tolerated)."""
    df = _store()
    df = pd.concat([df, pd.DataFrame(_rows("P", "5.5-5.75_f125", 125, "L5_L6", 1,
                                           seed=99))], ignore_index=True)
    m = make_curriculum_split(df, cells=CELLS_BASE + ["5.5-5.75_f125"], seed=0)
    assert m.groups["curriculum_conv_counts"]["5.5-5.75_f125"] == 1
    assert m.groups["curriculum_train_conv_counts"]["5.5-5.75_f125"] == 1


# --------------------------------------------------------------------------- #
# reached / unreached quarantine (methodology guard)
# --------------------------------------------------------------------------- #
FUTURE_CELL = "5-5.25_f117"


def test_unreached_cell_fully_quarantined() -> None:
    """A known cell the driver has NOT reached has every one of its (pre-merged)
    rows quarantined: never in train, never in the metric val fold, never in any
    per-cell eval holdout, and invisible to the trainer's curriculum weight-cap —
    it appears only under ``groups['quarantined_by_cell']``."""
    df = _store(extra_cell=True)
    # FUTURE_CELL is a known cell but the driver has only reached CELLS_BASE.
    m = make_curriculum_split(df, cells=CELLS_PLUS, seed=0, reached_cells=CELLS_BASE)
    future_rows = set(df[df["campaign"] == FUTURE_CELL]["record_id"].astype(str))
    assert future_rows                                   # sanity: rows exist

    # the whole cell is quarantined ...
    assert m.groups["quarantined_by_cell"][FUTURE_CELL] == sorted(future_rows)
    # ... out of BOTH folds ...
    assert future_rows.isdisjoint(set(m.train_ids))
    assert future_rows.isdisjoint(set(m.val_ids))
    # ... never scored by the honest gate, never a curriculum weight-cap campaign
    assert FUTURE_CELL not in m.groups["curriculum_val_by_cell"]
    assert FUTURE_CELL not in m.groups["curriculum_conv_counts"]
    assert FUTURE_CELL not in m.groups["cells"]

    # the reached cells are untouched: they keep their normal 80/20 holdouts.
    for cell in CELLS_BASE:
        assert m.groups["curriculum_val_by_cell"][cell]
        assert cell not in m.groups["quarantined_by_cell"]

    # folds partition the store MINUS the quarantine; quarantine closes the cover.
    all_ids = set(df["record_id"].astype(str))
    assert set(m.train_ids).isdisjoint(set(m.val_ids))
    assert set(m.train_ids) | set(m.val_ids) | future_rows == all_ids
    assert set(m.train_ids) | set(m.val_ids) == all_ids - future_rows


def test_quarantine_release_byte_identical_holdout() -> None:
    """When the future cell is reached, its stable-hash 80/20 split is byte-
    identical to a never-quarantined split — quarantine-earlier leaves no trace,
    and the per-cell holdout is independent of which OTHER cells are reached
    (growth invariance extends to quarantine release)."""
    df = _store(extra_cell=True)

    # reference: the future cell reached alongside everything (never quarantined).
    m_all = make_curriculum_split(df, cells=CELLS_PLUS, seed=0, reached_cells=CELLS_PLUS)
    ref_val = m_all.groups["curriculum_val_by_cell"][FUTURE_CELL]
    n_conv = m_all.groups["curriculum_conv_counts"][FUTURE_CELL]
    assert ref_val                                       # a real holdout
    assert m_all.groups["curriculum_train_conv_counts"][FUTURE_CELL] >= 0.70 * n_conv
    assert len(ref_val) <= 0.30 * n_conv + 1
    assert m_all.groups["quarantined_by_cell"] == {}     # nothing held

    # the future cell reached while EVERY other cell is quarantined -> its 80/20
    # membership is byte-identical (per-cell holdout is a pure function of the
    # cell's own record_ids, unaffected by the others' reached/quarantine state).
    m_iso = make_curriculum_split(df, cells=CELLS_PLUS, seed=0, reached_cells=[FUTURE_CELL])
    assert m_iso.groups["curriculum_val_by_cell"][FUTURE_CELL] == ref_val
    assert FUTURE_CELL not in m_iso.groups["quarantined_by_cell"]
    for c in CELLS_BASE:
        assert c in m_iso.groups["quarantined_by_cell"]

    # a prior quarantined split leaves NO trace: releasing == never-quarantined.
    m_quar = make_curriculum_split(df, cells=CELLS_PLUS, seed=0, reached_cells=CELLS_BASE)
    assert FUTURE_CELL in m_quar.groups["quarantined_by_cell"]
    m_released = make_curriculum_split(df, cells=CELLS_PLUS, seed=0, reached_cells=CELLS_PLUS)
    assert m_released.to_dict() == m_all.to_dict()


def test_reached_none_equals_all_reached_preserves_behavior() -> None:
    """``reached_cells=None`` (no reached/unreached distinction) is byte-identical
    to marking every known cell reached — the pre-quarantine behavior is preserved
    exactly when nothing is held back (existing all-reached store unchanged)."""
    df = _store(extra_cell=True)
    m_none = make_curriculum_split(df, cells=CELLS_PLUS, seed=0)
    m_all = make_curriculum_split(df, cells=CELLS_PLUS, seed=0, reached_cells=CELLS_PLUS)
    assert m_none.to_dict() == m_all.to_dict()
    assert m_none.groups["quarantined_by_cell"] == {}
    # every store row is still covered by the two folds (no quarantine hole).
    assert set(m_none.train_ids) | set(m_none.val_ids) == set(df["record_id"].astype(str))


# --------------------------------------------------------------------------- #
# honest no-regression gate: both champions scored on identical held-out ids
# --------------------------------------------------------------------------- #
class _Prediction:
    def __init__(self, mean: np.ndarray) -> None:
        self.mean = mean
        self.epistemic_std = np.full_like(mean, 0.1)
        self.calibrated_std = np.full_like(mean, 0.1)


class _RecordingModel:
    """Records every pattern it was asked to predict.  ``perfect`` recomputes the
    store truth from the pattern (Spearman ~+1); ``reversed`` predicts the negated
    truth (Spearman ~-1) — a rank-destroying regression with defined Spearman."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.seen: list[str] = []

    def predict(self, patterns, cases):
        self.seen = [p.canonical() for p in patterns]
        n = len(patterns)
        mean = np.zeros((n, 7), dtype=float)
        for i, p in enumerate(patterns):
            cyc, fr = _truth(p.canonical())
            if self.mode == "perfect":
                mean[i, 3], mean[i, 0] = cyc, fr
            else:                                   # rank-destroying (anti-correlated)
                mean[i, 3], mean[i, 0] = -cyc, -fr
        return _Prediction(mean)


def test_score_no_regression_cell_scores_identical_ids() -> None:
    from lpopt.curriculum import score_no_regression_cell
    df = _store()
    cell = "5.25-5.5_f117"
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    val_ids = m.groups["curriculum_val_by_cell"][cell]
    indexed = df.drop_duplicates("record_id").set_index("record_id")
    sub = indexed.loc[val_ids].reset_index()

    old = _RecordingModel("perfect")
    new = _RecordingModel("perfect")
    rids, rows = score_no_regression_cell(old, new, sub)

    # both champions saw the SAME held-out patterns, in the same order
    assert old.seen == new.seen
    assert len(old.seen) == len(sub)
    # the scored ids are exactly the cell's val holdout (out-of-sample), disjoint
    # from the train fold
    assert set(rids) == set(val_ids)
    assert set(rids).isdisjoint(set(m.train_ids))
    # two identical perfect models -> zero drop on every SCORED target
    scored_rows = [r for r in rows if r["drop"] is not None]
    assert scored_rows and all(abs(r["drop"]) < 1e-9 for r in scored_rows)
    # …and the flatness guards this slice cannot score are REPORTED, not absent:
    # the synthetic store carries no node_peak / map_cov column, so an unscored
    # guard must say so rather than quietly leave the family one axis short.
    unavail = {r["target"]: r["unavailable"] for r in rows if r.get("unavailable")}
    assert set(unavail) == {"node_peak", "map_cov"}
    assert all("node_peak" in v or "map_cov" in v for v in unavail.values())


def test_score_no_regression_cell_detects_true_drop() -> None:
    from lpopt.curriculum import score_no_regression_cell
    df = _store()
    cell = "5.25-5.5_f109"
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    val_ids = m.groups["curriculum_val_by_cell"][cell]
    indexed = df.drop_duplicates("record_id").set_index("record_id")
    sub = indexed.loc[val_ids].reset_index()
    _rids, rows = score_no_regression_cell(_RecordingModel("perfect"),
                                           _RecordingModel("reversed"), sub)
    # champion perfect (Spearman ~1), candidate rank-destroying -> a real drop
    drops = [r["drop"] for r in rows if r["drop"] is not None]
    assert drops and max(drops) > 0.5


# --------------------------------------------------------------------------- #
# the retrain-promotion gate guards the FLATNESS axes too (flatness-first)
#
# THE defect: ``NOREG_TARGETS`` was ``(cyclen, f_r)``, so promotion to champion —
# one level ABOVE the campaign — was still judged on cyclen and F_r skill with no
# flatness axis at all.  A candidate that had lost map-head skill could be
# promoted and then STEER a flat_power campaign, which is F_r selecting the model
# that selects the loading patterns.
#
# F_r is deliberately NOT dropped here: this gate guards the SAFETY path (the D1
# licensing gate flat_power still applies to every row), not the objective.
# --------------------------------------------------------------------------- #
def _flat_truth(canon: str) -> tuple[float, float]:
    """Deterministic (node_peak, map_cov) truth, independent of (cyclen, f_r)."""
    h = int(hashlib.sha1(("flat:" + canon).encode()).hexdigest(), 16)
    return 1.30 + (h % 1000) / 5000.0, 0.24 + ((h // 1000) % 1000) / 20000.0


def _with_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat = [_flat_truth(str(p)) for p in out["pattern"].tolist()]
    out["node_peak"] = [f[0] for f in flat]
    out["map_cov"] = [f[1] for f in flat]
    return out


class _FlatModel(_RecordingModel):
    """``_RecordingModel`` plus a MAP head.

    ``flat`` selects the map head's behaviour independently of the surrogate
    columns, so a candidate can be perfect on cyclen / F_r and still have lost
    node_peak / map_cov skill — the exact case the old gate promoted.
    """

    def __init__(self, mode: str, flat: str | None = None) -> None:
        super().__init__(mode)
        self.flat = flat or mode

    def predict_map_flatness(self, patterns, case, cell=0.0):
        pk, cv = [], []
        for p in patterns:
            peak, cov = _flat_truth(p.canonical())
            if self.flat == "perfect":
                pk.append(peak)
                cv.append(cov)
            else:                                  # rank-destroying
                pk.append(-peak)
                cv.append(-cov)
        pk = np.asarray(pk, dtype=float)
        cv = np.asarray(cv, dtype=float)
        return pk, np.full_like(pk, 0.01), cv, np.full_like(cv, 0.01)


def _flat_slice():
    df = _with_flat_columns(_store())
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    val_ids = m.groups["curriculum_val_by_cell"]["5.25-5.5_f109"]
    indexed = df.drop_duplicates("record_id").set_index("record_id")
    return indexed.loc[val_ids].reset_index()


def test_the_noreg_family_guards_the_flatness_axes() -> None:
    """The SCORED family is cyclen + f_r + node_peak + map_cov; the ENFORCING
    family drops f_r by default (D11, 2026-07-26) — scored, never a veto."""
    from lpopt.curriculum import NOREG_TARGETS, enforced_noreg_targets

    assert [t[0] for t in NOREG_TARGETS] == ["cyclen", "f_r", "node_peak", "map_cov"]
    assert enforced_noreg_targets() == ("cyclen", "node_peak", "map_cov")


def test_a_flatness_only_regression_is_caught_by_the_promotion_gate() -> None:
    """THE bug: a candidate that keeps cyclen/F_r skill and LOSES map-head skill.

    Under the old (cyclen, f_r) family every scored drop was ~0 and the model was
    promoted — then it steered a flat_power campaign on the axes it had just got
    worse at.
    """
    from lpopt.curriculum import score_no_regression_cell

    sub = _flat_slice()
    _rids, rows = score_no_regression_cell(
        _FlatModel("perfect", flat="perfect"),
        _FlatModel("perfect", flat="reversed"), sub)
    by = {r["target"]: r for r in rows}
    # cyclen and F_r are untouched — the old family would have seen nothing.
    assert abs(by["cyclen"]["drop"]) < 1e-9
    assert abs(by["f_r"]["drop"]) < 1e-9
    # …and the flatness axes catch it.
    assert by["node_peak"]["drop"] > 0.5
    assert by["map_cov"]["drop"] > 0.5
    assert by["node_peak"]["n"] >= 3


def test_f_r_is_still_scored_after_its_demotion() -> None:
    """F_r is DORMANT, not deleted (D11, 2026-07-26).

    The axis lost its promotion veto because the gate's own holdout carries no
    sub-1.55 labels and the in-band label ceiling is 0.839 — not because the
    quantity stopped mattering.  ``score_no_regression_cell`` must therefore keep
    measuring it, so the number is on the record the day the guard is switched
    back on (see ``FR_GUARD_ACTIVATION_CRITERIA``).
    """
    from lpopt.curriculum import score_no_regression_cell

    sub = _flat_slice()
    _rids, rows = score_no_regression_cell(
        _FlatModel("perfect", flat="perfect"),
        _FlatModel("reversed", flat="perfect"), sub)
    by = {r["target"]: r for r in rows}
    assert by["f_r"]["drop"] > 0.5
    # …and the flatness axes, untouched here, report no regression.
    assert abs(by["node_peak"]["drop"]) < 1e-9


def test_a_model_without_a_map_head_degrades_honestly() -> None:
    """No map head -> the flatness guards are REPORTED unavailable, not skipped."""
    from lpopt.curriculum import score_no_regression_cell

    sub = _flat_slice()                        # the slice HAS the labels…
    _rids, rows = score_no_regression_cell(    # …the models have no map head.
        _RecordingModel("perfect"), _RecordingModel("perfect"), sub)
    by = {r["target"]: r for r in rows}
    for name in ("node_peak", "map_cov"):
        assert by[name]["drop"] is None
        assert "no map head" in by[name]["unavailable"]


def test_gate_no_regression_names_the_axes_it_could_not_judge() -> None:
    """An unjudged guard must not read as a passed guard."""
    from lpopt.curriculum import gate_no_regression

    df = _with_flat_columns(_store())
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    val_by_cell = m.groups["curriculum_val_by_cell"]
    cells = ["5.25-5.5_f109"]
    out = gate_no_regression(_RecordingModel("perfect"), _RecordingModel("perfect"),
                             df, val_by_cell, cells, epsilon=0.05)
    assert out["pass"] is True                     # nothing measured regressed…
    assert out["guarded_targets"] == ["cyclen", "node_peak", "map_cov"]
    axes = {c["target"] for c in out["unavailable"]}
    assert axes == {"node_peak", "map_cov"}        # …but the report says which
    assert "UNJUDGED guarded axes" in out["note"]  #     guards never ran.
    assert "NOT verified regression-free" in out["note"]

    # with a map head on both sides nothing is unavailable; the only note left is
    # the REPORT-ONLY one for the demoted f_r axis (D11) — never silence.
    ok = gate_no_regression(_FlatModel("perfect"), _FlatModel("perfect"),
                            df, val_by_cell, cells, epsilon=0.05)
    assert ok["pass"] is True and ok["unavailable"] == []
    assert "UNJUDGED" not in ok["note"]
    assert "REPORT-ONLY axes (scored, NOT enforced): f_r" in ok["note"]

    # …and a flatness-only regression FAILS the gate.
    bad = gate_no_regression(_FlatModel("perfect", flat="perfect"),
                             _FlatModel("perfect", flat="reversed"),
                             df, val_by_cell, cells, epsilon=0.05)
    assert bad["pass"] is False
    assert bad["worst_drop"] > 0.5


def test_gate_coverage_stamp_reports_partial_guard_measurement() -> None:
    """A guard measured in ONE cell must not read like a guard measured in all.

    Regression test for the 2026-07-26 gate-blindness audit.  On the live store
    the promotion gate scored 36 curriculum cells and judged the two PRIMARY
    flatness axes in exactly 1 of them (only 2 of 36 cells carried any map label,
    and one of those held 2 rows, below the 3-row bar).  A total flatness collapse
    across the other 35 cells was demonstrably invisible: ``pass`` is a reduction
    over checks that produced a ``drop``, so an axis with no labels contributes
    nothing and cannot fail, while ``guarded_targets`` still advertised all three.

    Per the user decision the coverage deficit WARNS and never blocks — ``pass``
    keeps its exact former meaning — so what is pinned here is that the deficit is
    reported, quantified, and impossible to read as full coverage.
    """
    from lpopt.curriculum import gate_no_regression

    # Map labels on ONE cell only — the production shape, in miniature.
    df = _with_flat_columns(_store())
    blind_cell = df["campaign"].astype(str) == "5.25-5.5_f109"
    df.loc[blind_cell, ["node_peak", "map_cov"]] = float("nan")

    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    vbc = m.groups["curriculum_val_by_cell"]
    out = gate_no_regression(_FlatModel("perfect"), _FlatModel("perfect"),
                             df, vbc, list(CELLS_BASE), epsilon=0.05)

    assert out["pass"] is True                      # semantics UNCHANGED: warn only
    assert out["cells_scored"] == 2
    assert out["guarded_targets"] == ["cyclen", "node_peak", "map_cov"]

    # …but the measured truth is now beside the declaration and disagrees with it.
    gm = out["guarded_measured"]
    assert gm["cyclen"]["cells"] == 2               # every cell carries a cyclen
    assert gm["node_peak"]["cells"] == 1            # only the labelled one
    assert gm["map_cov"]["cells"] == 1
    assert out["blind_targets"] == []               # measured somewhere, just not everywhere
    assert "GUARD COVERAGE" in out["note"]
    assert "node_peak 1/2 cells" in out["note"]

    # An axis measured NOWHERE is named outright.
    df2 = _with_flat_columns(_store())
    df2[["node_peak", "map_cov"]] = float("nan")
    m2 = make_curriculum_split(df2, cells=CELLS_BASE, seed=0)
    out2 = gate_no_regression(_FlatModel("perfect"), _FlatModel("perfect"),
                              df2, m2.groups["curriculum_val_by_cell"],
                              list(CELLS_BASE), epsilon=0.05)
    assert out2["pass"] is True                     # still never blocks
    assert out2["blind_targets"] == ["map_cov", "node_peak"]
    assert "BLIND guarded axes" in out2["note"]
    assert "says nothing whatever about them" in out2["note"]


def test_unscoreable_scalar_axis_is_reported_not_dropped() -> None:
    """The one genuinely silent path: a SCALAR axis that cannot be scored.

    ``_unavailable`` used to be called only under ``if map_head``, so an
    unscoreable cyclen produced no row at all — absent from ``checks``,
    ``unavailable``, ``note`` and the console alike.  Unreached in production only
    because cyclen carries a label on 100% of rows, which is a property of today's
    data rather than a guarantee.
    """
    from lpopt.curriculum import gate_no_regression

    df = _with_flat_columns(_store())
    df["cyclen"] = float("nan")                     # scalar axis becomes unscoreable
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    out = gate_no_regression(_FlatModel("perfect"), _FlatModel("perfect"),
                             df, m.groups["curriculum_val_by_cell"],
                             list(CELLS_BASE), epsilon=0.05)

    axes = {c["target"] for c in out["unavailable"]}
    assert "cyclen" in axes                         # it reports itself now
    assert out["blind_targets"] == ["cyclen"]
    assert "BLIND guarded axes: cyclen" in out["note"]


# --------------------------------------------------------------------------- #
# F_r DEMOTED to report-only (user decision 2026-07-26)
#
# The previous round guarded ``f_r`` here "because it guards the safety path".
# That is an argument about the axis's IMPORTANCE; a promotion gate is a
# REGRESSION detector, and gating on F_r today vetoes models for a DATA
# limitation rather than a model regression:
#
#   * the gate's own slice has NO decision-band labels — measured 2026-07-26,
#     zero of the 36 done cells' val holdouts (1,592 rows) carry a single
#     ``f_r < 1.55`` row; the lowest val F_r anywhere is 1.5974, so the "safety
#     guard" scored bulk F_r rank two-tenths above the licensing limit; and
#   * in the band where it WOULD matter the transpose-pair label ceiling is
#     rho_max = 0.839 (transpose_noise_measured_20260725.md §2.2) — a perfect
#     physics model cannot beat 0.84 there.
#
# Core F_r is set by the hottest ASSEMBLY, so the axis only acquires learnable
# signal once FA-optimized assemblies are loaded.  ``f_r`` is therefore DORMANT,
# not removed: still scored, still printed, still reported as unenforced — and
# one config setting away from being a guard again.
# --------------------------------------------------------------------------- #
class _AxisModel(_FlatModel):
    """Per-axis control: ``cyclen``, ``f_r`` and the map head are independently
    perfect or rank-destroying, so a regression can be confined to ONE axis —
    which is exactly what distinguishes "F_r cannot veto" from "nothing vetoes".
    """

    def __init__(self, cyclen: str = "perfect", f_r: str = "perfect",
                 flat: str = "perfect") -> None:
        super().__init__("perfect", flat=flat)
        self.cyclen_mode, self.fr_mode = cyclen, f_r

    def predict(self, patterns, cases):
        self.seen = [p.canonical() for p in patterns]
        mean = np.zeros((len(patterns), 7), dtype=float)
        for i, p in enumerate(patterns):
            cyc, fr = _truth(p.canonical())
            mean[i, 3] = cyc if self.cyclen_mode == "perfect" else -cyc
            mean[i, 0] = fr if self.fr_mode == "perfect" else -fr
        return _Prediction(mean)


_FR_CELLS = ["5.25-5.5_f109"]


def _fr_gate(old, new, **kw):
    from lpopt.curriculum import gate_no_regression
    df = _with_flat_columns(_store())
    m = make_curriculum_split(df, cells=CELLS_BASE, seed=0)
    return gate_no_regression(old, new, df, m.groups["curriculum_val_by_cell"],
                              _FR_CELLS, epsilon=0.05, **kw)


def test_an_f_r_only_regression_no_longer_blocks_promotion() -> None:
    """THE reversal: a candidate that lost ONLY F_r rank skill is promoted.

    Under the previous (guarded) family this returned ``pass=False``.  The drop
    is still measured and still reported — the axis lost its veto, not its score.
    """
    out = _fr_gate(_AxisModel(), _AxisModel(f_r="reversed"))

    assert out["pass"] is True                             # no veto…
    fr = [c for c in out["checks"] if c["target"] == "f_r"]
    assert fr and fr[0]["drop"] > 0.5                      # …but SCORED, and big
    assert fr[0]["enforced"] is False
    # ``worst_drop`` keeps the ``pass == (worst_drop <= epsilon)`` invariant by
    # tracking the ENFORCED axes; the unenforced excursion is not hidden, it is
    # reported separately.
    assert out["worst_drop"] <= out["epsilon"]
    assert out["worst_drop_any_axis"] > 0.5
    assert out["guarded_targets"] == ["cyclen", "node_peak", "map_cov"]
    assert out["report_only_targets"] == ["f_r"]
    assert out["scored_targets"] == ["cyclen", "f_r", "node_peak", "map_cov"]


def test_a_pass_can_never_be_read_as_f_r_verified() -> None:
    """Honest reporting: the gate says in words that F_r was scored, exceeded
    epsilon, and was admitted anyway — reusing the same ``note`` channel the
    unavailable-axis machinery already writes to."""
    out = _fr_gate(_AxisModel(), _AxisModel(f_r="reversed"))

    note = out["note"]
    assert "REPORT-ONLY axes (scored, NOT enforced): f_r" in note
    assert "does NOT mean" in note and "regression-free" in note
    assert "exceeded epsilon and was ADMITTED anyway" in note
    assert "gate_noreg_fr_guard_enabled" in note           # …and how to change it
    assert out["fr_guard"]["enforced"] is False

    # a clean candidate still says F_r had no teeth, without the ADMITTED clause.
    clean = _fr_gate(_AxisModel(), _AxisModel())
    assert "REPORT-ONLY axes (scored, NOT enforced): f_r" in clean["note"]
    assert "ADMITTED anyway" not in clean["note"]
    assert clean["pass"] is True


def test_the_activation_switch_restores_the_f_r_veto_in_one_setting() -> None:
    """The FA-optimized phase enables the guard with one config setting — no code
    change.  Same inputs, ``fr_guarded=True`` (``[curriculum]
    gate_noreg_fr_guard_enabled``), and the veto is back."""
    out = _fr_gate(_AxisModel(), _AxisModel(f_r="reversed"), fr_guarded=True)

    assert out["pass"] is False
    assert out["worst_drop"] > 0.5
    assert out["guarded_targets"] == ["cyclen", "f_r", "node_peak", "map_cov"]
    assert out["report_only_targets"] == []
    assert out["fr_guard"]["enforced"] is True
    assert "REPORT-ONLY" not in out.get("note", "")
    fr = [c for c in out["checks"] if c["target"] == "f_r"]
    assert fr and fr[0]["enforced"] is True


def test_demoting_f_r_did_not_disarm_the_other_guards() -> None:
    """cyclen and the flatness axes keep their veto — the demotion is one axis
    wide, not a general softening of the gate."""
    assert _fr_gate(_AxisModel(), _AxisModel(cyclen="reversed"))["pass"] is False
    assert _fr_gate(_AxisModel(), _AxisModel(flat="reversed"))["pass"] is False
    # …and a candidate that regressed on F_r AND cyclen still fails, on cyclen.
    both = _fr_gate(_AxisModel(), _AxisModel(cyclen="reversed", f_r="reversed"))
    assert both["pass"] is False


def test_the_f_r_activation_criteria_are_recorded_in_code() -> None:
    """(3) of the work order: the criteria live in code so nobody re-derives them.

    They are also currently UNMET by construction — the measured band ceiling
    (0.839) is below the bar (0.95) — which is why the default is off.
    """
    from lpopt import curriculum as C

    assert len(C.FR_GUARD_ACTIVATION_CRITERIA) == 3
    joined = " ".join(C.FR_GUARD_ACTIVATION_CRITERIA)
    assert "FA-optimized" in joined                       # (a)
    assert str(C.FR_GUARD_MIN_BAND_LABELS_PER_CELL) in joined   # (b)
    assert str(C.FR_GUARD_BAND_HI) in joined
    assert str(C.FR_GUARD_MIN_LABEL_CEILING) in joined    # (c)
    # The band is the D1 in-loop safety gate — where the model's F_r PREDICTION
    # adjudicates — not the D2 licensing constant 1.55, which is a compliance
    # column no in-loop decision is taken at.
    assert C.FR_GUARD_BAND_HI == 1.70
    assert C.FR_GUARD_LICENSING_LIMIT == 1.55
    assert C.FR_GUARD_MIN_BAND_LABELS_PER_CELL == 30      # = the sigma0 calibration n
    # criterion (c) is not met today: 0.839 measured vs a 0.95 bar.
    assert C.FR_GUARD_MEASURED_LABEL_CEILING == 0.839
    assert C.FR_GUARD_MEASURED_LABEL_CEILING < C.FR_GUARD_MIN_LABEL_CEILING
    assert C.FR_GUARD_KNOB == "[curriculum] gate_noreg_fr_guard_enabled"
    assert C.enforced_noreg_targets() == ("cyclen", "node_peak", "map_cov")
    assert C.enforced_noreg_targets(fr_guarded=True) == (
        "cyclen", "f_r", "node_peak", "map_cov")


def test_the_gate_measures_criterion_b_on_its_own_slice() -> None:
    """Criterion (b) is MEASURED every run, not asserted once: the gate counts the
    decision-band labels in the very holdout it scored, so the decision to flip
    the switch is made against data.  The synthetic slice reproduces the real
    store's shape — band rows exist, but nowhere near the 30/cell the epsilon
    calibration needs."""
    g = _fr_gate(_AxisModel(), _AxisModel())["fr_guard"]

    assert g["target"] == "f_r" and g["band_hi"] == 1.70
    assert g["cells_scored"] == 1
    assert set(g["band_labels_by_cell"]) == set(_FR_CELLS)
    n_band = g["band_labels_by_cell"][_FR_CELLS[0]]
    assert 0 < n_band < g["min_band_labels_per_cell"]
    assert g["cells_meeting_label_criterion"] == 0
    assert g["measured_label_ceiling"] < g["min_label_ceiling"]
    assert len(g["activation_criteria"]) == 3


def test_the_fr_guard_knob_is_threaded_not_orphaned() -> None:
    """A config knob nothing reads is a dead knob.  Both promotion paths — the
    curriculum driver's validate_gate and the ``lpopt gate-promote`` CLI — must
    read it, or the FA-optimized phase would flip it and see nothing change."""
    import inspect

    from lpopt.cli import cmd_gate_promote
    from lpopt.curriculum import CurriculumDriver

    for fn in (CurriculumDriver._gate_no_regression, cmd_gate_promote):
        src = inspect.getsource(fn)
        assert "gate_noreg_fr_guard_enabled" in src, fn.__qualname__
        assert "fr_guarded=" in src, fn.__qualname__


def test_the_activation_band_is_measured_where_the_prediction_adjudicates() -> None:
    """LOW (3): criteria (b)/(c) used to be measured at the D2 LICENSING constant
    1.55, but the model's F_r prediction adjudicates at the D1 in-loop SAFETY
    gate 1.70 (``FlatPowerSpec.fr_limit`` — the binary veto ``flat_power`` applies
    to every candidate).  A criterion measured off the decision surface is not a
    criterion about the decision, so the band is the D1 gate; 1.55 stays as a
    reported licensing reference.
    """
    from lpopt import curriculum as C
    from lpopt.search.acquisition import FlatPowerSpec

    assert C.FR_GUARD_BAND_HI == 1.70                 # where the prediction decides
    assert C.FR_GUARD_BAND_HI == FlatPowerSpec().fr_limit
    assert C.FR_GUARD_LICENSING_LIMIT == 1.55         # D2, reported not measured on
    joined = " ".join(C.FR_GUARD_ACTIVATION_CRITERIA)
    assert "1.7" in joined and "D1" in joined
    # the band choice is justified in the comment, not merely asserted
    import inspect
    src = inspect.getsource(C)
    assert "adjudicates" in src


def test_the_fr_guard_block_reports_both_bands() -> None:
    """The gate reports the D1 decision-band count (criterion (b)) AND the D2
    licensing-band count, so flipping the switch is argued against both."""
    g = _fr_gate(_AxisModel(), _AxisModel())["fr_guard"]

    assert g["band_hi"] == 1.70
    assert g["licensing_limit"] == 1.55
    assert set(g["band_labels_by_cell"]) == set(_FR_CELLS)
    assert set(g["licensing_band_labels_by_cell"]) == set(_FR_CELLS)
    # the licensing band is a strict subset of the decision band
    assert (g["licensing_band_labels_by_cell"][_FR_CELLS[0]]
            <= g["band_labels_by_cell"][_FR_CELLS[0]])
    assert g["cells_meeting_label_criterion"] == 0


def test_the_early_return_still_carries_the_gate_keys() -> None:
    """LOW (5): the "no previous cells" path returned a bare dict, so a consumer
    reading ``guarded_targets`` / ``fr_guard`` off a gate.json got a KeyError on
    the very first cell — and no record that F_r was deferred on that run."""
    from lpopt.curriculum import gate_no_regression

    out = gate_no_regression(_AxisModel(), _AxisModel(), _with_flat_columns(_store()),
                             {}, [], epsilon=0.05)

    assert out["pass"] is True and out["checks"] == []
    assert out["guarded_targets"] == ["cyclen", "node_peak", "map_cov"]
    assert out["report_only_targets"] == ["f_r"]
    assert out["scored_targets"] == ["cyclen", "f_r", "node_peak", "map_cov"]
    assert out["fr_guard"]["enforced"] is False
    assert out["fr_guard"]["knob"] == "[curriculum] gate_noreg_fr_guard_enabled"
    assert out["epsilon"] == 0.05
    assert out["fr_guard"]["cells_scored"] == 0
    # and the same shape with the guard armed
    on = gate_no_regression(_AxisModel(), _AxisModel(), _with_flat_columns(_store()),
                            {}, [], epsilon=0.05, fr_guarded=True)
    assert on["guarded_targets"] == ["cyclen", "f_r", "node_peak", "map_cov"]
    assert on["fr_guard"]["enforced"] is True
