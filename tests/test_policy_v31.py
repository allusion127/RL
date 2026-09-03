"""Guards for the policy v3.1 code deltas (track B).

``data/reports/policy_v31_prereg_20260831_DRAFT.md`` §9a-H lists seven
assertions.  The four this track owns are here, on a SYNTHETIC corpus so the
guards run without the 111-column parquet and cannot be quietly satisfied by a
lucky property of the real data:

* **(a)** v3's 51 scalars are bit-identical inside v3.1's 53, and exactly two
  columns are added — the §4c structural selection rule, not a p-value;
* **(b)** the stage-2 branch IS stage 1 at initialisation, so clause 5
  (``fr`` parent-blocked AUC >= 0.678) holds structurally rather than by luck;
* **(d)** the held-out cell reaches none of the K cross-fit folds, and
  **(e)** no ``val`` component reaches the gate pool — the new §3d discipline,
  without which lambda selection and judgement would read the same rows.  The
  assignment is emitted as a file so the metrics are computed on a hashable
  artefact rather than on a re-derivation;
* **(f)** the two burnt columns are nonzero on a rewire and EXACTLY zero on a
  move that relocates no burnt assembly;
* **(g)** serving reproduces training on the logit scale and the Platt map is
  order-preserving.

Plus the guard the flags themselves need.  Every v3.1 flag defaults off, and the
combinations that would LOOK like a v3.1 run while being a v3 run are refused
rather than silently ignored — ``--lam-grid`` without ``--stage2 on`` is the one
that matters, because the listwise term exists only inside stage 2.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

import mine_policy_corpus as M
from lpopt.data.schema import pack_pattern, unpack_pattern
from lpopt.policy.v3 import (
    BURNT_ABSMOV_R_UNIT, BURNT_ABSMOV_UNIT, FOLDS_V31, NEW_SCALARS_V3,
    NEW_SCALARS_V31, POLICY_SCHEMA_V3, POLICY_SCHEMA_V31, PROSPECTIVE_CELL,
    PROSPECTIVE_CELL_V31, XFIT_K, assert_serving_parity_v31, build_splits_v31,
    calib_index, platt_serve, scalar_features_v3, scalar_features_v31,
    xfit_indices,
)
from lpopt.search.genome import GenomeError, random_genome
from lpopt.vendor.masterrl.domain import FuelItem, Pattern

_PAIR = "A1_A2"


# --------------------------------------------------------------------------- #
# a synthetic 111-column corpus
# --------------------------------------------------------------------------- #
def _board(seed: int) -> str:
    return pack_pattern(random_genome(random.Random(seed), _PAIR, 30).to_pattern())


#: Orbit units that own exactly one slot.  Editing a whole unit is what keeps a
#: hand-built board decodable: ``GeneralOrbitGenome.from_pattern`` rejects an
#: axis-twin pair whose two arms disagree, so a single-slot edit on an axis unit
#: would make the fixture, not the code under test, the thing that failed.
_INTERIOR = [u[0] for u in M.UNIT_SLOTS if len(u) == 1]


def _rewire(packed: str, seed: int = 0) -> str:
    """Relocate two BURNT assemblies and change nothing else.

    This is the move class §4b measured blind: the fresh multiset, every fresh
    slot and therefore all twelve v3 lattice move-channels are untouched, and
    only the burnt sub-lattice moves.  The two source cards trade places, so the
    multiset of sources is preserved and the genome's consumption invariant
    still holds.
    """
    items = list(unpack_pattern(packed).items)
    burnt = [i for i in _INTERIOR if not items[i].is_fresh]
    rng = random.Random(seed)
    for _ in range(400):
        a, b = rng.sample(burnt, 2)
        # Purely GEOMETRIC admissibility, computed WITHOUT calling the
        # descriptors under test.  Writing ``A = mult_a * (|r_a - r_prev_b| -
        # |r_a - r_prev_a|)`` and ``B`` symmetrically, the two features are
        # ``A + B`` and ``r_a A + r_b B``; both vanish only when ``r_a == r_b``
        # or ``A == B == 0``.  Excluding those two cases here is what makes the
        # 100%-liveness assertion below a real assertion rather than a hope.
        pa = M._prev_slot(items[a].x, items[a].y)
        pb = M._prev_slot(items[b].x, items[b].y)
        if pa is None or pb is None:
            continue
        ra, rb = M.SLOT_RADIUS[a], M.SLOT_RADIUS[b]
        rpa, rpb = M.SLOT_RADIUS[pa], M.SLOT_RADIUS[pb]
        if ra == rb or abs(ra - rpa) == abs(ra - rpb):
            continue
        swapped = list(items)
        # the SOURCE card moves; the destination's positional rotation stays
        swapped[a] = FuelItem(kind="shuffle", restart=items[b].restart,
                              x=items[b].x, y=items[b].y,
                              rotation=items[a].rotation)
        swapped[b] = FuelItem(kind="shuffle", restart=items[a].restart,
                              x=items[a].x, y=items[a].y,
                              rotation=items[b].rotation)
        out = pack_pattern(Pattern(tuple(swapped)))
        if out == packed:
            continue
        try:                                   # still a decodable core?
            M.genome_of(out)
        except GenomeError:
            continue
        return out
    raise AssertionError("could not build a burnt relocation on this board")


def _rewire_equal_radius(packed: str, seed: int = 0) -> str:
    """Exchange two burnt assemblies between slots of EQUAL radius and multiplicity.

    The blind spot §4c-(ii) does not cover, built on purpose.  Writing
    ``f(x) = |r - x|`` for the shared destination radius ``r`` and ``m`` for the
    shared multiplicity, the swap contributes ``m[f(p_b) - f(p_a)] +
    m[f(p_a) - f(p_b)] = 0`` to ``burnt_absmov`` and ``r`` times that to
    ``burnt_absmov_r``: both vanish identically, for any ``g(r)``, because every
    purely radial moment cancels by symmetry.  The two assemblies really did
    move — ``burnt_slots_moved == 2`` — which is the whole content of the miss.
    """
    items = list(unpack_pattern(packed).items)
    burnt = [i for i in _INTERIOR if not items[i].is_fresh]
    rng = random.Random(seed)
    for _ in range(600):
        a, b = rng.sample(burnt, 2)
        pa = M._prev_slot(items[a].x, items[a].y)
        pb = M._prev_slot(items[b].x, items[b].y)
        if pa is None or pb is None or pa == pb:
            continue
        if M.SLOT_RADIUS[a] != M.SLOT_RADIUS[b]:
            continue
        if M.SLOT_MULT[a] != M.SLOT_MULT[b]:
            continue
        if M.SLOT_RADIUS[pa] == M.SLOT_RADIUS[pb]:
            continue                            # then nothing moved radially
        swapped = list(items)
        swapped[a] = FuelItem(kind="shuffle", restart=items[b].restart,
                              x=items[b].x, y=items[b].y,
                              rotation=items[a].rotation)
        swapped[b] = FuelItem(kind="shuffle", restart=items[a].restart,
                              x=items[a].x, y=items[a].y,
                              rotation=items[b].rotation)
        out = pack_pattern(Pattern(tuple(swapped)))
        if out == packed:
            continue
        try:
            M.genome_of(out)
        except GenomeError:
            continue
        return out
    raise AssertionError("could not build an equal-radius burnt exchange")


def _fresh_flip(packed: str, seed: int = 0) -> str:
    """Relabel one FRESH unit; every ``S`` token stays exactly where it was.

    The control for the guard below: a move that touches only the fresh
    sub-lattice must leave both v3.1 columns identically zero.
    """
    items = list(unpack_pattern(packed).items)
    fresh = [i for i in _INTERIOR if items[i].is_fresh]
    labels = sorted({items[i].batch for i in fresh})
    rng = random.Random(seed)
    for _ in range(400):
        i = rng.choice(fresh)
        other = [b for b in labels if b != items[i].batch]
        if not other:
            break
        out_items = list(items)
        out_items[i] = FuelItem(kind="fresh", batch=rng.choice(other),
                                rotation=items[i].rotation)
        out = pack_pattern(Pattern(tuple(out_items)))
        if out == packed:
            continue
        try:
            M.genome_of(out)
        except GenomeError:
            continue
        return out
    raise AssertionError("could not build a fresh-only move on this board")


def _record(rid: str, parent: str | None, packed: str, library: str,
            f_xy: float, **over) -> dict:
    row = {
        "record_id": rid, "parent_record_id": parent, "pattern": packed,
        "case_pair": _PAIR, "feed": 121, "library_id": library,
        "campaign": "unit", "dataset": "P", "generator": "unit",
        "converged": True, "valid": True, "f_r": 1.50, "f_q": 2.0,
        "cbc_max": 1200.0, "ao_abs": 0.05, "cyclen": 500.0, "node_peak": 1.3,
        "map_cov": 0.2, "f_xy": f_xy,
    }
    row.update(over)
    return row


def _steps(n_parents: int = 40, per_parent: int = 4, seed: int = 7
           ) -> pd.DataFrame:
    """A REAL 111-column corpus over synthetic boards.

    ``mine_policy_corpus.build_steps`` is called rather than hand-written, so
    every one of v3's 107 columns is the column the miner actually emits and a
    guard here cannot pass because the fixture invented a convenient value.  The
    four v3.1 columns are then appended by the function under test.

    Shape: two cells (so the held-out rule has something to hold out), and a
    chain ``p -> c0 -> {c1, c2, c3}`` per parent so ``_components`` sees a real
    four-row component instead of four singletons.  Odd children are burnt
    relocations (``rewire_*``), even children are fresh-only moves.
    """
    rng = np.random.default_rng(seed)
    store = []
    for p in range(n_parents):
        parent = _board(1000 + p)
        library = "ga80" if p % 2 else "paramA"
        store.append(_record(f"p{p}", None, parent, library, 1.60))
        prev = (f"p{p}", parent)
        for c in range(per_parent):
            child = (_rewire(prev[1], seed=p * 17 + c) if c % 2 else
                     _fresh_flip(prev[1], seed=p * 17 + c))
            store.append(_record(f"c{p}_{c}", prev[0], child, library,
                                 1.60 + float(rng.normal(0.0, 0.02))))
            if c == 0:
                prev = (f"c{p}_{c}", child)
    steps = M.build_steps(pd.DataFrame(store), {}, {})
    steps = steps.reset_index(drop=True)
    steps["era_current"] = True
    steps["interventional"] = True
    burnt = M.burnt_move_columns(steps)
    return pd.concat([steps, burnt], axis=1)


@pytest.fixture(scope="module")
def steps() -> pd.DataFrame:
    return _steps()


@pytest.fixture(scope="module")
def cells(steps: pd.DataFrame) -> list[str]:
    return sorted(steps["cell"].unique())


# --------------------------------------------------------------------------- #
# 1. the corpus columns — §4c / §4d, and test (f)
# --------------------------------------------------------------------------- #
def test_the_corpus_gains_exactly_four_columns_and_two_are_features() -> None:
    assert M.BURNT_MOVE_COLUMNS == (
        "burnt_absmov", "burnt_absmov_r", "burnt_slots_moved",
        "burnt_token_complete")
    assert len(M.V3_SCHEMA_COLUMNS) == 22 and len(M.V31_SCHEMA_COLUMNS) == 4
    # 85 (v2) -> 107 (v3) -> 111 (v3.1); the two diagnostics never become
    # features, exactly as ``gd_table_complete`` never did in v3.
    assert set(NEW_SCALARS_V31) < set(M.BURNT_MOVE_COLUMNS)
    assert len(NEW_SCALARS_V31) == 2
    assert not set(NEW_SCALARS_V31) & {
        "burnt_slots_moved", "burnt_token_complete"}


def test_the_rejected_conserving_columns_are_absent() -> None:
    """§4c: ``rew_cor_r2`` / ``rew_flux_*`` fail the STRUCTURAL rule.

    They are conserving moments — expressible as ``sum mult * g(r) * X`` — so
    they are excluded regardless of any readout.  ``rew_cor_r2``'s own p = 0.0103
    did not clear its Bonferroni line 0.0023 either, but that is not why it is
    out, and a later round must not be able to re-admit it by improving the
    p-value.
    """
    banned = {"rew_cor_r2", "burnt_cor_r2", "d_burnt_cor_r2", "rew_flux_r",
              "rew_flux_r2"}
    assert not banned & set(M.BURNT_MOVE_COLUMNS)
    assert not banned & set(NEW_SCALARS_V31)


def test_burnt_columns_are_nonzero_on_a_rewire_and_zero_without_one(
        steps: pd.DataFrame) -> None:
    """§9a-H(f), and the registered liveness rule of §4c.

    Both columns are the parent->child DIFFERENCE of a board moment, so a move
    that relocates no burnt assembly is EXACTLY zero and needs no separate
    liveness flag; a rewire that does relocate burnt is live.

    **What the 100% assertion below does and does not say.**  ``_rewire``
    excludes the degenerate class on purpose (``ra == rb``, and the
    ``|ra - rpa| == |ra - rpb|`` cancellation), so on generated rows the pair is
    live on all of them and the assertion is tight.  It is NOT a claim about the
    registered corpus, and the two frames give DIFFERENT numbers, so each is
    labelled with the frame it was measured on (``mine_policy_corpus.py --v31``
    prints the second):

    * ``steps_v3.parquet`` — 214 labelled current-era same-cell rewire rows,
      ``burnt_absmov`` live on 73.8%, ``burnt_absmov_r`` on 96.7%, the pair
      96.7%; over all 381 such rewire rows, 73.0 / 95.0 / 95.0.
    * ``steps_v31.parquet`` (the corpus the round reads, +81 r2 edges) — 267
      labelled rows, 74.9 / 97.4 / 97.4; over all 434 rows, 73.7 / 95.6 / 95.6.

    §4c-(ii) as worded — 100% per column — admits NEITHER column on EITHER
    frame, and the pair reading the round ships is a relaxation adopted after
    the measurement.  The class that misses is pinned separately, and
    deliberately, by
    :func:`test_the_equal_radius_exchange_is_the_registered_blind_spot`.
    """
    rewire = steps["move_class"].str.startswith("rewire").to_numpy()
    fresh_only = ~rewire
    assert rewire.any() and fresh_only.any(), "the fixture built only one class"
    for col in NEW_SCALARS_V31:
        v = steps[col].to_numpy(float)
        assert np.all(np.isfinite(v))
        assert np.all(v[fresh_only] == 0.0), (
            f"{col} moved on a fresh-only move: no burnt token changed, so the "
            f"difference of the board moments must be identically 0, which is "
            f"why §4c needs no separate liveness flag")

    # Liveness is a property of the PAIR, and that is not a convenience.  A swap
    # of two burnt assemblies can leave ``burnt_absmov`` at zero while moving
    # ``burnt_absmov_r``: writing ``A`` and ``B`` for the two slots' contributions
    # the columns are ``A + B`` and ``r_a A + r_b B``, so ``A = -B`` kills the
    # first and (whenever ``r_a != r_b``) not the second.  Both die together only
    # if the move relocated nothing.  §4c admits BOTH columns for exactly this
    # reason, and the tolerance is float noise on a difference of two sums.
    a = steps["burnt_absmov"].to_numpy(float)
    ar = steps["burnt_absmov_r"].to_numpy(float)
    live = (np.abs(a) > 1e-12) | (np.abs(ar) > 1e-12)
    assert live[rewire].all(), (
        f"the burnt family is dead on "
        f"{int((~live[rewire]).sum())}/{int(rewire.sum())} burnt relocations; "
        f"§4c admits a column only if it is live on 100% of rewire rows")
    assert not live[fresh_only].any()
    assert (steps.loc[fresh_only, "burnt_slots_moved"] == 0).all()
    assert (steps.loc[rewire, "burnt_slots_moved"] > 0).all()
    assert steps["burnt_token_complete"].all()


def test_the_corpus_is_107_columns_plus_exactly_four(steps: pd.DataFrame) -> None:
    """§4d's arithmetic: 85 (v2) -> 107 (v3) -> 111 (v3.1)."""
    base = [c for c in steps.columns
            if c not in (*M.BURNT_MOVE_COLUMNS, "era_current", "interventional")]
    assert len(base) == 107, len(base)
    assert len(base) + len(M.BURNT_MOVE_COLUMNS) == 111


def test_the_v3_lattice_channels_are_blind_to_exactly_these_rows(
        steps: pd.DataFrame) -> None:
    """§4b: the deficit the two columns fill, stated as the reason they exist.

    On a burnt relocation every move-level v3 lattice channel is identically
    zero — that is the 197-row, 7-block hole ``nc2_E1b_rawzero.csv`` measured —
    and the two v3.1 columns are the only within-parent resolution those rows
    have.  The registered ground for admitting them is this COVERAGE, not an
    effect size (§4c).
    """
    rewire = steps["move_class"].str.startswith("rewire").to_numpy()
    blind = ("d_fresh_gd_mass", "d_fresh_gd_r_center", "d_fresh_gd_share_periph",
             "d_fresh_gdwt_mass", "d_fresh_kinf0_mass", "d_fresh_kinf0_r_center",
             "n_fresh_type_changed", "fresh_gd_contrast")
    for col in blind:
        v = np.nan_to_num(steps[col].to_numpy(float))
        assert np.all(v[rewire] == 0.0), f"{col} is not blind on a rewire"
    assert steps.loc[rewire, "fresh_type_multiset_changed"].eq(False).all()


def test_the_burnt_columns_are_not_conserved_by_a_pure_permutation() -> None:
    """The (i) half of the §4c selection rule, as an executable statement.

    A pure relocation of two equally-burnt assemblies conserves every
    ``sum mult * g(r) * X`` moment — that is why v3's twelve lattice channels
    are identically zero on 197 rewire rows — and must NOT conserve these two.
    """
    parent = _board(4242)
    child = _rewire(parent, seed=11)
    out = M.burnt_slot_move(parent, child)
    assert out["burnt_slots_moved"] == 2
    assert (abs(out["burnt_absmov"]) > 1e-12
            or abs(out["burnt_absmov_r"]) > 1e-12)
    # the fresh sub-lattice really is untouched, i.e. this is the blind spot
    assert M.fresh_type_move(parent, child, None)["n_fresh_type_changed"] == 0.0


def test_the_equal_radius_exchange_is_the_registered_blind_spot() -> None:
    """§4c-(ii) is NOT met as worded, and this is the class that misses it.

    Exchanging two burnt assemblies between destination slots of EQUAL radius
    and EQUAL multiplicity moves two assemblies and leaves BOTH features exactly
    zero: writing ``f(x) = |r - x|`` for the shared radius and ``m`` for the
    shared multiplicity, the two contributions are ``m[f(p_b) - f(p_a)]`` and
    ``m[f(p_a) - f(p_b)]``, which cancel in ``burnt_absmov`` and, because both
    carry the SAME ``g(r)``, in ``burnt_absmov_r`` too.  No choice of ``g``
    repairs it — every purely radial moment cancels by symmetry — so this is a
    property of the registered descriptor family and not a bug to be fixed.

    It is pinned as a test rather than only disclosed in prose so that the
    measured shortfall (on ``steps_v3.parquet``: the pair live on 96.7% of the
    214 labelled current-era same-cell rewire rows, the 7 misses all of this
    class) cannot be quietly re-described later as a data accident, and so that
    a future edit which makes the columns nonzero here shows up as a FAILING
    test demanding a prereg amendment rather than as a silent redefinition of
    the two registered features.  ``burnt_slots_moved`` is live on the class,
    which is why it is the diagnostic and not a third feature (§4d admits
    exactly two).
    """
    parent = _board(4242)
    child = _rewire_equal_radius(parent, seed=5)
    out = M.burnt_slot_move(parent, child)
    assert out["burnt_slots_moved"] == 2
    assert out["burnt_token_complete"] is True
    assert out["burnt_absmov"] == pytest.approx(0.0, abs=1e-12)
    assert out["burnt_absmov_r"] == pytest.approx(0.0, abs=1e-12)
    # ... and the fresh sub-lattice is untouched, so v3 is blind here too: the
    # row has NO within-parent resolution at all in either feature contract.
    assert M.fresh_type_move(parent, child, None)["n_fresh_type_changed"] == 0.0


def test_the_burnt_descriptors_are_antisymmetric_and_self_zero() -> None:
    parent, child = _board(51), None
    child = _rewire(parent, seed=3)
    fwd = M.burnt_slot_move(parent, child)
    rev = M.burnt_slot_move(child, parent)
    for col in NEW_SCALARS_V31:
        assert fwd[col] == pytest.approx(-rev[col], abs=1e-12)
    same = M.burnt_slot_move(parent, parent)
    assert same["burnt_absmov"] == 0.0 and same["burnt_absmov_r"] == 0.0
    assert same["burnt_slots_moved"] == 0.0


def test_an_unparseable_shuffle_token_is_flagged_not_imputed() -> None:
    """``burnt_token_complete`` exists so a parse failure cannot read as "no move"."""
    parent = _board(77)
    tokens = parent.split("|")
    i = next(j for j, t in enumerate(tokens) if t.startswith("S:"))
    broken = list(tokens)
    broken[i] = "S:1:Z:99:2"                    # off the canonical quarter
    out = M.burnt_slot_move("|".join(broken), parent)
    assert out["burnt_token_complete"] is False
    ok = M.burnt_slot_move(parent, parent)
    assert ok["burnt_token_complete"] is True


def test_the_v31_mode_refuses_to_overwrite_the_v3_corpus(tmp_path) -> None:
    """§4d: ``steps_v3.parquet`` is byte-preserved, whatever the flags say."""
    import types

    path = tmp_path / "steps_v3.parquet"
    args = types.SimpleNamespace(
        steps_v3=path, out_v31=path, merge_campaign="", apply=True, force=False,
        store=None, sa_lineage=tmp_path / "nope.parquet")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        M.cmd_v31(args)


# --------------------------------------------------------------------------- #
# 2. the feature vector — §9a-H(a)
# --------------------------------------------------------------------------- #
def test_v31_features_strictly_contain_v3s_bit_for_bit(
        steps: pd.DataFrame) -> None:
    v3, n3 = scalar_features_v3(steps)
    v31, n31 = scalar_features_v31(steps)
    assert set(n3) < set(n31)
    assert sorted(set(n31) - set(n3)) == sorted(NEW_SCALARS_V31)
    assert len(n31) == len(n3) + 2
    for name in n3:
        np.testing.assert_array_equal(
            v3[:, n3.index(name)], v31[:, n31.index(name)],
            err_msg=f"{name} changed inside the v3.1 vector")
    # and v3's own twelve are still all there, unrenamed
    assert set(NEW_SCALARS_V3) <= set(n31)


def test_the_burnt_scales_are_registered_constants_not_fitted(
        steps: pd.DataFrame) -> None:
    """§10-4: one constant for every fold, so normalization leaks nothing."""
    _, names = scalar_features_v31(steps)
    half = steps.head(len(steps) // 2)
    a = scalar_features_v31(steps)[0][:len(half), names.index("burnt_absmov")]
    b = scalar_features_v31(half)[0][:, names.index("burnt_absmov")]
    np.testing.assert_array_equal(a, b)
    assert BURNT_ABSMOV_UNIT > 0 and BURNT_ABSMOV_R_UNIT > 0


def test_v31_features_refuse_a_v3_corpus(steps: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="burnt"):
        scalar_features_v31(steps.drop(columns=list(NEW_SCALARS_V31)))


# --------------------------------------------------------------------------- #
# 3. cross-fit splits — §3a / §3d, tests (d) and (e)
# --------------------------------------------------------------------------- #
def test_the_held_out_cell_is_in_no_cross_fit_fold(
        steps: pd.DataFrame, cells: list[str]) -> None:
    """§9a-H(d).  The transfer cell is opened ONCE, after the gate exists."""
    splits = build_splits_v31(steps, holdout_cell=cells[0])
    held = (steps["cell"] == cells[0]).to_numpy()
    assert held.any()
    assert (splits.loc[held, "fold"] == "prospective_cell").all()
    assert (splits.loc[held, "xfit_fold"] == -1).all()
    for k in range(XFIT_K):
        idx = xfit_indices(splits, k)
        for part in ("train", "val", "eval"):
            assert not held[idx[part]].any(), (
                f"the held-out cell reached fold {k}'s {part} set")


def test_no_val_component_reaches_the_gate_pool(
        steps: pd.DataFrame, cells: list[str]) -> None:
    """§9a-H(e).  lambda is selected on ``val``; the gate must not read it."""
    from lpopt.policy.data import _components

    splits = build_splits_v31(steps, holdout_cell=cells[0])
    comp = pd.Series(_components(steps), index=steps.index)
    val = set(comp[splits["fold"] == "val"])
    pool = set(comp[splits["fold"] == "pool"])
    assert val and pool, "the fixture produced no val or no pool component"
    assert not (val & pool), sorted(val & pool)
    assert (splits.loc[splits["fold"] == "val", "xfit_fold"] == -1).all()
    assert set(calib_index(splits)) == set(
        np.flatnonzero((splits["fold"] == "pool").to_numpy()))


def test_no_val_component_reaches_the_pool_on_a_MIXED_era_frame(
        steps: pd.DataFrame, cells: list[str]) -> None:
    """The regime §9a-H(e) can actually fail in, made reachable.

    On an all-current-era frame the per-era ``val`` draw sees the whole pool at
    once and the exclusion is trivially satisfied.  The hole is a lineage chain
    that CROSSES the era boundary: derived per era subset it is two components
    and can be dealt to ``val`` on one side and to the gate pool on the other,
    while any check that re-derives components the same way sees two innocent
    components.  Here every chain straddles, and the assertion is made with the
    components of the WHOLE frame.
    """
    from lpopt.policy.data import _components

    mixed = steps.copy()
    legacy = mixed["child_record_id"].astype(str).str.endswith("_1").to_numpy()
    mixed.loc[legacy, "era_current"] = False
    assert legacy.any() and (~legacy).any()

    comp = pd.Series(_components(mixed), index=mixed.index)
    era = mixed["era_current"].to_numpy(bool)
    straddling = {k for k in comp.unique()
                  if len(set(era[(comp == k).to_numpy()])) == 2}
    assert straddling, "the fixture built no era-crossing lineage component"

    splits = build_splits_v31(mixed, holdout_cell=cells[0])
    val = set(comp[splits["fold"] == "val"])
    pool = set(comp[splits["fold"] == "pool"])
    assert val, "no val component was drawn, so this guard checked nothing"
    assert not (val & pool), sorted(val & pool)
    # and a val component is val in BOTH eras, not just in the era it was drawn
    for key in val:
        assert set(splits.loc[(comp == key).to_numpy(), "fold"]) <= {
            "val", "prospective_cell"}


def test_no_lineage_component_straddles_two_cross_fit_blocks(
        steps: pd.DataFrame, cells: list[str]) -> None:
    from lpopt.policy.data import _components

    splits = build_splits_v31(steps, holdout_cell=cells[0])
    pool = (splits["fold"] == "pool").to_numpy()
    comp = _components(steps)[pool]
    block = splits.loc[pool, "xfit_fold"].to_numpy()
    for key in set(comp):
        assert len(set(block[comp == key])) == 1, (
            f"component {key} was dealt to more than one block; the leakage "
            f"guard is the COMPONENT, not the parent")


def test_the_blocks_partition_the_pool_and_every_row_is_scored_once(
        steps: pd.DataFrame, cells: list[str]) -> None:
    splits = build_splits_v31(steps, holdout_cell=cells[0])
    pool = np.flatnonzero((splits["fold"] == "pool").to_numpy())
    seen = np.concatenate([xfit_indices(splits, k)["eval"] for k in range(XFIT_K)])
    assert sorted(seen) == sorted(pool)
    assert len(set(seen)) == len(seen), "a pool row was scored out-of-fold twice"
    for k in range(XFIT_K):
        idx = xfit_indices(splits, k)
        assert not set(idx["train"]) & set(idx["eval"])
        assert not set(idx["train"]) & set(idx["val"])


def test_the_split_is_deterministic_and_the_folds_are_the_registered_four(
        steps: pd.DataFrame, cells: list[str]) -> None:
    a = build_splits_v31(steps, seed=20260903, holdout_cell=cells[0])
    b = build_splits_v31(steps, seed=20260903, holdout_cell=cells[0])
    pd.testing.assert_frame_equal(a, b)
    assert set(a["fold"]) <= set(FOLDS_V31)
    assert PROSPECTIVE_CELL_V31 != PROSPECTIVE_CELL, (
        "§3c replaced the transfer cell; v3's failed three of the four "
        "registered difficulty conditions")
    with pytest.raises(ValueError, match="at least 2"):
        build_splits_v31(steps, k=1, holdout_cell=cells[0])


# --------------------------------------------------------------------------- #
# 4. serving parity and the Platt map — §5c / §9a-H(g)
# --------------------------------------------------------------------------- #
def test_the_platt_map_is_monotone_and_widens_without_reordering() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(-2.0, 3.0, 400)
    out = assert_serving_parity_v31(z, z.copy(), a=0.35, b=1.2)
    assert out["max_abs_logit_gap"] == 0.0
    assert out["logit_p90_p10"] > out["prob_p90_p10"]     # the §5c artefact
    p = platt_serve(z, a=0.35, b=1.2)
    assert np.array_equal(np.argsort(z), np.argsort(p))
    with pytest.raises(ValueError, match="positive"):
        platt_serve(z, a=-1.0, b=0.0)
    with pytest.raises(ValueError, match="positive"):
        platt_serve(z, a=0.0, b=0.0)


def test_serving_parity_fails_loudly_on_a_feature_contract_drift() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=64)
    drift = z.copy()
    drift[3] += 1e-3
    with pytest.raises(AssertionError, match="does not reproduce"):
        assert_serving_parity_v31(z, drift)
    with pytest.raises(AssertionError, match="rows"):
        assert_serving_parity_v31(z, z[:10])


def test_the_two_serving_contracts_are_distinct() -> None:
    assert POLICY_SCHEMA_V31 != POLICY_SCHEMA_V3
    assert POLICY_SCHEMA_V31 == "policy_move_v31"


def test_the_platt_fit_cannot_return_a_reordering_map() -> None:
    """§5c: monotonicity is a property of the PARAMETRIZATION, not of the data.

    The slope is carried as ``exp(alpha)``, so there is no sample — separable,
    noisy, or adversarially anti-correlated — on which the fit can return
    ``a <= 0`` and reorder the candidates.  The anti-correlated case is the one
    that matters: a naive unconstrained logistic fit answers it with a NEGATIVE
    slope, which would silently invert every ranking statistic in §5a while
    every parity check on logits still passed.
    """
    from lpopt.policy.v3 import fit_platt

    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 2.0, 500)
    for y in (  # separable, noisy, and anti-correlated with the logit
            (z > 0).astype(float),
            (rng.random(500) < 1.0 / (1.0 + np.exp(-(0.4 * z - 1.0)))).astype(float),
            (z < 0).astype(float)):
        a, b = fit_platt(z, y)
        assert a > 0.0 and np.isfinite(a) and np.isfinite(b)
        p = platt_serve(z, a=a, b=b)
        # NON-DECREASING, not strictly increasing: a separable fold drives the
        # slope high enough that the sigmoid saturates to exactly 0.0 / 1.0 in
        # float64, so distinct logits can share a served probability.  That is a
        # loss of RESOLUTION, which clause 4's width readout measures, and not a
        # reordering, which is what §5c forbids -- demanding strict order here
        # would be a test of float64's dynamic range.
        assert np.all(np.diff(p[np.argsort(z, kind="stable")]) >= 0.0)
    # and it really is a FIT: a separable fold buys a steep slope, not a > 0 by
    # construction with the data ignored
    assert fit_platt(z, (z > 0).astype(float))[0] > 1.0

    with pytest.raises(ValueError, match="single label value"):
        fit_platt(z, np.ones_like(z))
    with pytest.raises(ValueError, match="labels"):
        fit_platt(z, np.ones(7))
    with pytest.raises(ValueError, match="empty"):
        fit_platt(z[:0], z[:0])
    with pytest.raises(ValueError, match="non-finite"):
        fit_platt(np.array([0.0, np.nan]), np.array([0.0, 1.0]))


def test_the_platt_map_is_fitted_on_the_calib_fold_and_nothing_else(
        steps: pd.DataFrame, cells: list[str]) -> None:
    """§5c: ``val`` selects lambda and the holdout is opened once — neither may
    reach the calibration, and the guard is that the fold is taken from the
    SPLIT rather than from the caller."""
    from lpopt.policy.v3 import fit_platt, fit_platt_v31

    splits = build_splits_v31(steps, holdout_cell=cells[0])
    rng = np.random.default_rng(5)
    z = rng.normal(size=len(splits))
    y = (rng.random(len(splits)) < 0.3).astype(float)
    out = fit_platt_v31(z, y, splits)

    idx = calib_index(splits)
    assert out["n_calib"] == len(idx) < len(splits), (
        "the calibration read the whole corpus, so val and the held-out cell "
        "reached the map")
    assert out["calib_base_rate"] == pytest.approx(y[idx].mean())
    a, b = fit_platt(z[idx], y[idx])
    assert (out["a"], out["b"]) == pytest.approx((a, b))
    # poisoning the rows OUTSIDE calib must not move the fitted map by a digit
    poisoned = y.copy()
    other = np.setdiff1d(np.arange(len(splits)), idx)
    poisoned[other] = 1.0 - poisoned[other]
    assert other.size, "the fixture left no rows outside the calib fold"
    again = fit_platt_v31(z, poisoned, splits)
    assert (again["a"], again["b"]) == pytest.approx((out["a"], out["b"]))

    with pytest.raises(ValueError, match="one logit per corpus row"):
        fit_platt_v31(z[:-1], y, splits)


def test_serving_parity_refuses_a_mis_stamped_checkpoint() -> None:
    """§9a-H(g): a v3 checkpoint served through the v3.1 path reproduces itself.

    Which is exactly why the logit comparison alone could never catch the
    mis-stamp this hook exists for; the stamp is checked before the logits are.
    """
    z = np.linspace(-2.0, 2.0, 32)
    ok = {"policy_schema": POLICY_SCHEMA_V31,
          "scalar_names": [f"s{i}" for i in range(53)]}
    assert assert_serving_parity_v31(z, z.copy(), meta=ok)["max_abs_logit_gap"] == 0.0

    with pytest.raises(AssertionError, match="not 'policy_move_v31'"):
        assert_serving_parity_v31(z, z.copy(),
                                  meta={"policy_schema": POLICY_SCHEMA_V3})
    with pytest.raises(AssertionError, match="stamp and the"):
        assert_serving_parity_v31(z, z.copy(), meta={
            "policy_schema": POLICY_SCHEMA_V31,
            "scalar_names": [f"s{i}" for i in range(51)]})


# --------------------------------------------------------------------------- #
# 5. the listwise teacher and the stage-2 branch — §2b, tests (b) and (c)
# --------------------------------------------------------------------------- #
def test_the_teacher_is_raw_gain_masked_by_feasibility(steps: pd.DataFrame) -> None:
    from lpopt.policy.train_v3 import TEACHER_EPS, listwise_teacher

    frame = steps.copy()
    frame.loc[frame.index[:2], "d_cyclen"] = -50.0      # infeasible: cyclen loss
    q = listwise_teacher(frame)
    assert np.all(q[:2] == 0.0), (
        "an infeasible candidate carries no teacher mass; it stays in the "
        "student's softmax denominator instead (§2b)")
    parents = frame["parent_record_id"].astype(str).to_numpy()
    for key in pd.unique(parents):
        idx = np.flatnonzero(parents == key)
        if q[idx].sum() > 0:
            assert q[idx].sum() == pytest.approx(1.0)
    # the ordering the teacher induces is the raw gain's, not the clipped one's
    key = pd.unique(parents)[-1]
    idx = np.flatnonzero((parents == key) & (q > 0))
    if len(idx) >= 2:
        u = -frame["d_f_xy"].to_numpy(float)[idx]
        assert np.array_equal(np.argsort(u), np.argsort(q[idx]))
    assert 0.0 < TEACHER_EPS < 1.0


def test_the_teacher_rejects_a_degenerate_temperature(steps: pd.DataFrame) -> None:
    from lpopt.policy.train_v3 import listwise_teacher

    with pytest.raises(ValueError, match="temperature"):
        listwise_teacher(steps, temp=0.0)
    with pytest.raises(ValueError, match="smoothing"):
        listwise_teacher(steps, eps=1.0)


def test_the_teacher_temperature_is_the_registered_clip_constant() -> None:
    """§2b: stage 2 introduces NO new free constant."""
    from lpopt.policy.train_v3 import TEACHER_TEMP
    from lpopt.policy.v3 import TARGET_CLIP_V3

    assert TEACHER_TEMP == TARGET_CLIP_V3["fxy"] == 0.060


def test_the_listwise_term_is_zero_for_a_perfect_ranker_at_eps_zero() -> None:
    torch = pytest.importorskip("torch")
    from lpopt.policy.train_v3 import listwise_ce

    q = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    good = torch.tensor([50.0, -50.0, -50.0, 50.0, -50.0])
    bad = torch.tensor([-50.0, 50.0, 50.0, -50.0, 50.0])
    sizes = [3, 2]
    assert float(listwise_ce(good, q, sizes)) < 1e-6
    assert float(listwise_ce(bad, q, sizes)) > 10.0


def test_stage_two_is_stage_one_at_initialisation(steps: pd.DataFrame) -> None:
    """§9a-H(b): clause 5 must hold STRUCTURALLY, not by luck."""
    torch = pytest.importorskip("torch")
    from lpopt.policy.net import PolicyNet, PolicyNetConfig
    from lpopt.policy.train_v3 import Stage2FxyBranch, assert_stage2_init_is_stage1

    cfg = PolicyNetConfig(arm="cnn", in_channels=4, n_cond=9, width=8,
                          n_blocks=2, groups=4, head_hidden=16, n_heads=3)
    torch.manual_seed(0)
    base = PolicyNet(cfg)
    branch = Stage2FxyBranch(base)
    cells = torch.rand(5, 4, 19, 19)
    cells[:, 0] = (cells[:, 0] > 0.3).float()
    cond = torch.randn(5, 9)
    assert_stage2_init_is_stage1(branch, cells, cond)

    # the trunk is frozen and the fr/flat logits stay the frozen tensor even
    # after the branch has moved
    with torch.no_grad():
        for p in branch.branch.parameters():
            p.add_(torch.randn_like(p))
    branch.eval()
    with torch.no_grad():
        got, want = branch(cells, cond), base(cells, cond)
    assert torch.equal(got[:, :2], want[:, :2])
    assert not torch.equal(got[:, 2], want[:, 2])
    assert all(not p.requires_grad for p in branch.base.parameters())
    assert all(p.requires_grad for p in branch.branch.parameters())
    with pytest.raises(AssertionError, match="seeded from the frozen"):
        assert_stage2_init_is_stage1(branch, cells, cond)


def _toy_cache(frame: pd.DataFrame):
    """A stand-in :class:`PatternCache`; the board CONTENT is not what is tested."""
    from lpopt.policy.data import PatternCache

    pats = sorted(set(frame["parent_pattern"]) | set(frame["child_pattern"]))
    rng = np.random.default_rng(0)
    return PatternCache(
        index={p: i for i, p in enumerate(pats)},
        slots=rng.random((len(pats), 4, 69)).astype(np.float16),
        globals_=rng.random((len(pats), 3)).astype(np.float32),
        mirror=np.arange(len(pats), dtype=np.int32),
        channels=[f"c{i}" for i in range(4)],
        globals_names=[f"g{i}" for i in range(3)])


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_the_stage_two_step_runs_end_to_end_and_moves_only_the_branch(
        steps: pd.DataFrame, lam: float) -> None:
    """The listwise branch, the parent batcher and the frozen head, together."""
    torch = pytest.importorskip("torch")
    from lpopt.policy.net import PolicyNet, PolicyNetConfig
    from lpopt.policy.train_v3 import train_stage2
    from lpopt.policy.v3 import PolicyStepsV3, weights_v3

    sub = steps.head(80).reset_index(drop=True)
    cache = _toy_cache(sub)
    scalars, names = scalar_features_v31(sub)
    weights = weights_v3(sub, pd.Series("train", index=sub.index))
    sets = {name: PolicyStepsV3(sub, cache, scalars, weights,
                                delta_channels=[0, 1], augment=False, seed=1)
            for name in ("train", "val")}
    assert sets["train"].n_cond == len(names) + 3

    torch.manual_seed(0)
    base = PolicyNet(PolicyNetConfig(
        arm="cnn", in_channels=sets["train"].n_channels,
        n_cond=sets["train"].n_cond, width=8, n_blocks=2, groups=4,
        head_hidden=16, n_heads=3))
    before = {k: v.clone() for k, v in base.state_dict().items()}

    model, meta = train_stage2(base, sets=sets, frames={"train": sub},
                               device="cpu", lam=lam, epochs=1, patience=1,
                               batch_size=32)
    assert meta["lam"] == lam and meta["teacher"] == "raw"
    assert meta["n_groups"] > 0, "the fixture built no multi-candidate parent"
    for k, v in base.state_dict().items():
        assert torch.equal(v, before[k]), f"stage 2 moved the frozen trunk at {k}"
    model.eval()
    with torch.no_grad():
        item = sets["train"][0]
        cells = torch.as_tensor(item["cells"])[None]
        cond = torch.as_tensor(item["cond"])[None]
        assert torch.equal(model(cells, cond)[:, :2], base(cells, cond)[:, :2])


def test_exactly_one_lambda_reaches_the_gate_and_it_is_picked_on_seed_means(
) -> None:
    """§9d.  Selecting inside the seed loop would let members of ONE ensemble
    carry different lambdas, and the gate would then average logits produced
    under different objectives while ``metrics.json`` reported a single
    ``stage2_lam_selected`` per member — there would be no one lambda that the
    gated artefact is.  The tie rule points at the SMALLER lambda because the
    smaller lambda is the smaller deviation from v3: a tie must not buy a
    listwise term.
    """
    from lpopt.policy.train_v3 import select_stage2_lambda

    # per-seed argmax would pick 1.0 (it wins on seed 1); the seed MEAN picks 0.3
    scores = {0.0: {1: 0.50, 2: 0.50, 3: 0.50},
              0.3: {1: 0.55, 2: 0.62, 3: 0.60},
              1.0: {1: 0.90, 2: 0.40, 3: 0.40}}
    assert select_stage2_lambda(scores) == 0.3
    assert max(scores[1.0].values()) > max(scores[0.3].values())

    tie = {0.0: {1: 0.7, 2: 0.5}, 0.3: {1: 0.5, 2: 0.7}}
    assert select_stage2_lambda(tie) == 0.0

    with pytest.raises(ValueError, match="grid was empty"):
        select_stage2_lambda({})
    with pytest.raises(ValueError, match="scored on no seed"):
        select_stage2_lambda({0.0: {}})
    with pytest.raises(ValueError, match="non-finite"):
        select_stage2_lambda({0.0: {1: float("nan")}})
    with pytest.raises(ValueError, match="same seeds"):
        select_stage2_lambda({0.0: {1: 0.5, 2: 0.5}, 0.3: {1: 0.6}})


def test_stage_two_refuses_the_registered_gain_teacher() -> None:
    """§2b: aligning the teacher with the gate is a MEASURED failure."""
    pytest.importorskip("torch")
    from lpopt.policy.train_v3 import train_stage2

    with pytest.raises(ValueError, match="RAW gain"):
        train_stage2(None, sets={}, frames={}, device="cpu", lam=0.0,
                     teacher="registered")


# --------------------------------------------------------------------------- #
# 6. the flags — the v3 path is bit-identical when they are off
# --------------------------------------------------------------------------- #
def _args(**over):
    from lpopt.policy.train_v3 import _parser

    return _parser().parse_args([*sum(([k, str(v)] for k, v in over.items()), [])])


def test_every_v31_flag_defaults_off_and_the_run_stamps_v3() -> None:
    from lpopt.policy.train_v3 import LAMBDA_GRID, assert_v3_path_untouched

    args = _args()
    assert args.stage2 == "off"
    assert args.xfit_k == 0
    assert tuple(args.lam_grid) == LAMBDA_GRID == (0.0, 0.3, 1.0)
    assert args.holdout_cell == PROSPECTIVE_CELL         # v3's, not v3.1's
    assert args.steps.endswith("steps_v3.parquet")
    assert args.splits == ""
    stamp = assert_v3_path_untouched(args)
    assert {k: stamp[k] for k in ("enabled", "version", "lam_grid", "xfit_k",
                                  "teacher", "teacher_temp", "teacher_eps",
                                  "crossfit")} == {
        "enabled": False, "version": "v3", "lam_grid": [0.0, 0.3, 1.0],
        "xfit_k": 0, "teacher": "raw", "teacher_temp": 0.060,
        "teacher_eps": 0.10, "crossfit": False}
    # prereg delta D is reported, never assumed: with no --splits the module
    # may be importable and the run is still not a cross-fit run.
    assert stamp["delta_d"]["available"] is False


def test_a_lambda_grid_without_stage_two_is_refused() -> None:
    """It would read as a listwise run in the log and be a v3 run."""
    from lpopt.policy.train_v3 import assert_v3_path_untouched

    with pytest.raises(SystemExit, match="without --stage2 on"):
        assert_v3_path_untouched(_args(**{"--lam-grid": "0,1"}))


def test_stage_two_alone_stays_refused_and_the_cross_fit_shape_needs_delta_D(
        tmp_path, monkeypatch) -> None:
    """§3a registers the cross-fit and the cell replacement as ONE change.

    ``--stage2 on`` alone would train on v3's SINGLE alternating split (37
    parents with >= 8 F_xy candidates, 16 live -- the row §3a marks REJECTED)
    and stamp the checkpoint v3.1.  That refusal is permanent for this round and
    is asserted first.

    ``--stage2 on --xfit-k 5`` was refused for a different reason: without the
    metric module the flag is still the split EMISSION and returns before a
    weight exists.  Prereg delta D (``lpopt.policy.metrics_v31``) lifts it, but
    ONLY together with ``--splits`` naming the frozen STEP 0-a assignment by its
    registered sha256 -- a module that would score any assignment is not a
    pre-registration.  All three legs are asserted so the lift cannot widen by
    accident: no ``--splits``, wrong bytes, and the registered bytes.
    """
    from lpopt.policy import metrics_v31 as m31
    from lpopt.policy.train_v3 import assert_v3_path_untouched

    with pytest.raises(SystemExit, match="without --xfit-k"):
        assert_v3_path_untouched(_args(**{"--stage2": "on",
                                          "--lam-grid": "0,1"}))
    # (a) the module is importable but no assignment was named
    with pytest.raises(SystemExit, match="--splits was not given"):
        assert_v3_path_untouched(_args(**{
            "--stage2": "on", "--xfit-k": 5,
            "--holdout-cell": PROSPECTIVE_CELL_V31}))
    # (b) an assignment that is not the registered one
    foreign = tmp_path / "splits_v31.csv"
    foreign.write_text("\n".join(["child_record_id,fold,xfit_fold",
                                  "x,pool,0", ""]))
    with pytest.raises(SystemExit, match="not the registered cross-fit"):
        assert_v3_path_untouched(_args(**{
            "--stage2": "on", "--xfit-k": 5, "--splits": str(foreign),
            "--holdout-cell": PROSPECTIVE_CELL_V31}))
    # (c) the registered bytes: the refusal lifts, and says so in the stamp
    monkeypatch.setattr(m31, "splits_fingerprint_ok",
                        lambda path, expected=None: str(path) == str(foreign))
    stamp = assert_v3_path_untouched(_args(**{
        "--stage2": "on", "--xfit-k": 5, "--splits": str(foreign),
        "--holdout-cell": PROSPECTIVE_CELL_V31}))
    assert stamp["enabled"] is True and stamp["crossfit"] is True
    assert stamp["version"] == "v31"
    assert stamp["delta_d"]["available"] is True
    assert stamp["delta_d"]["module"] == "lpopt.policy.metrics_v31"


def test_the_split_emission_stands_alone_but_only_on_the_registered_cell() -> None:
    """§3c: ``--holdout-cell`` defaults to v3's, and forgetting it is refused.

    ``--xfit-k`` is an EMISSION mode, not a training knob, so it stands alone —
    but the floor guard inside :func:`emit_crossfit_splits` only judges the
    realized live count on the REGISTERED holdout, so an emission that inherits
    v3's default cell writes an assignment on the wrong frame with the gate-pool
    floor silently disabled.  That is the failure this refusal exists for.
    """
    from lpopt.policy.train_v3 import assert_v3_path_untouched

    solo = assert_v3_path_untouched(_args(**{
        "--xfit-k": 5, "--holdout-cell": PROSPECTIVE_CELL_V31}))
    assert solo["xfit_k"] == 5 and solo["enabled"] is False
    with pytest.raises(SystemExit, match="§3c registers its held-out cell"):
        assert_v3_path_untouched(_args(**{"--xfit-k": 5}))   # v3's cell


def test_the_cross_fit_emission_writes_the_assignment_and_its_census(
        steps: pd.DataFrame, cells: list[str], tmp_path) -> None:
    """§3a: the assignment the metrics are computed on is a hashable artefact."""
    import json

    from lpopt.policy.train_v3 import emit_crossfit_splits

    census = emit_crossfit_splits(steps, tmp_path, k=XFIT_K, seed=20260903,
                                  holdout_cell=cells[0])
    assert (tmp_path / "splits_v31.csv").is_file()
    assert json.loads((tmp_path / "xfit_census.json").read_text()) == census
    assert len(census["blocks"]) == XFIT_K
    assert census["k"] == XFIT_K and census["holdout_cell"] == cells[0]
    assert census["calib_rows"] == census["folds"]["pool"]["n_rows"]
    written = pd.read_csv(tmp_path / "splits_v31.csv")
    assert list(written.columns) == ["child_record_id", "fold", "xfit_fold"]
    assert len(written) == len(steps)
    assert set(written["fold"]) <= set(FOLDS_V31)
    # every pool row is an eval row of exactly one block
    assert sum(b["eval"]["n_rows"] for b in census["blocks"]) ==         census["folds"]["pool"]["n_rows"]


def test_the_feature_vector_and_the_serving_stamp_come_out_of_one_call(
        steps: pd.DataFrame, monkeypatch) -> None:
    """The mis-stamp guard, exercised rather than only documented.

    The first cut of this track branched on the flag when writing
    ``policy_schema`` but not when featurizing, so ``--stage2 on`` produced a
    checkpoint stamped ``policy_move_v31`` carrying v3's 51 names: the two burnt
    columns reached no model and ``MoveScorerV31`` would have rendered 53 names
    against a 51-name checkpoint.  ``featurize_round`` closes that by taking the
    vector and the stamp through one door; here the door is forced by making the
    v3.1 featurizer hand back a v3 vector, and the run must die.
    """
    from lpopt.policy import train_v3 as T
    from lpopt.policy.v3 import N_SCALARS_V3, N_SCALARS_V31

    _, names31, new31 = T.featurize_round(steps, v31=True)
    assert len(names31) == N_SCALARS_V31 == 53
    assert set(NEW_SCALARS_V31) <= set(names31)
    assert set(NEW_SCALARS_V31) <= set(new31)

    _, names3, new3 = T.featurize_round(steps, v31=False)
    assert len(names3) == N_SCALARS_V3 == 51
    assert not set(NEW_SCALARS_V31) & set(names3)
    assert not set(NEW_SCALARS_V31) & set(new3)
    assert set(NEW_SCALARS_V3) <= set(new3)

    monkeypatch.setattr(T, "scalar_features_v31", scalar_features_v3)
    with pytest.raises(SystemExit, match="must carry the v3.1 feature contract"):
        T.featurize_round(steps, v31=True)


def test_the_emission_refuses_to_write_a_gate_pool_below_the_floor(
        steps: pd.DataFrame, tmp_path) -> None:
    """§3a/§6a: clause 2B's power is computed on the pool's live parent count,
    so a knob that spends it must show up as a refusal and not as a quietly
    weaker gate.  The floor is judged on the REGISTERED holdout only — on any
    other cell the number is a re-derivation on a different frame."""
    from lpopt.policy.train_v3 import emit_crossfit_splits
    from lpopt.policy.v3 import MIN_POOL_LIVE_V31

    assert MIN_POOL_LIVE_V31 > 0
    with pytest.raises(SystemExit, match="registered floor"):
        emit_crossfit_splits(steps, tmp_path, k=XFIT_K, seed=20260903,
                             holdout_cell=PROSPECTIVE_CELL_V31)
    assert not (tmp_path / "splits_v31.csv").exists(), (
        "the refused assignment was written anyway")
    # the synthetic frame has 4 candidates per parent, so it can never clear an
    # 8-candidate floor; on any OTHER holdout it is written without judgement
    emit_crossfit_splits(steps, tmp_path, k=XFIT_K, seed=20260903,
                         holdout_cell="no/such/cell")
    assert (tmp_path / "splits_v31.csv").is_file()


def test_the_registered_gain_teacher_is_not_reachable_from_the_cli() -> None:
    with pytest.raises(SystemExit):
        _args(**{"--teacher": "registered"})


def _launcher_args(*, v31: bool):
    import types

    return types.SimpleNamespace(
        v31=v31, seeds=3, base_seed=20260903, epochs=120, patience=15,
        batch_size=256, lr="1e-3", weight_decay="1e-4", width=112, n_blocks=6,
        protocol="revB", num_workers=8, extra="", xfit_k=5, lam_grid="0,0.3,1.0",
        teacher_temp="0.060", teacher_eps="0.10", stage2_lr="1e-4",
        stage2_epochs=40, ts="policy_v31")


def test_the_launcher_renders_the_registered_v31_training_command() -> None:
    """The v3 round's command must not gain a character from this track.

    And the v3.1 round now has a TRAINING command, because prereg delta D
    landed.  What is asserted is that it is the §9c command and not a
    lookalike: the frozen split is passed BY PATH (so ``--xfit-k`` is the
    consumption of a hashed assignment and not the emission), the registered
    lambda grid is passed whole (§9d: it is selected on ``val``, not swept as
    arms), and the teacher is the raw one §2b registered.
    """
    import train_policy_v3 as L

    v3 = L.train_args(_launcher_args(v31=False))
    assert "steps_v3.parquet" in v3 and "--stage2" not in v3
    assert "N1_N2/f113/ga80" in v3 and "_feature_cache_v3.npz" in v3
    assert "policy_v3_v2_baseline.csv" in v3 and "--xfit-k" not in v3
    assert "--splits" not in v3

    cmd = L.train_args(_launcher_args(v31=True))
    assert "--stage2 on" in cmd
    assert "--xfit-k 5" in cmd and f"--splits {L.SPLITS_V31}" in cmd
    assert L.SPLITS_V31 == "data/policy/v31_split/splits_v31.csv"
    assert "--lam-grid 0,0.3,1.0" in cmd
    assert "--teacher raw" in cmd and "--teacher-temp 0.060" in cmd
    assert "steps_v31.parquet" in cmd and "_feature_cache_v31.npz" in cmd
    assert f'--holdout-cell "{PROSPECTIVE_CELL_V31}"' in cmd
    assert "policy_v31_v2_baseline.csv" in cmd
    assert "steps_v3.parquet " not in cmd        # the frozen corpus is untouched

    # the emission step still renders, and still trains nothing
    xfit = L.xfit_command(_launcher_args(v31=True))
    assert "--xfit-k 5" in xfit and "--stage2" not in xfit
    assert "--splits" not in xfit
    rnd = L._round(_launcher_args(v31=True))
    assert rnd["steps"] == L.STEPS_V31
    assert rnd["cache"] == "data/policy/_feature_cache_v31.npz"
    assert rnd["baseline"] == L.V2_BASELINE_V31
    assert L.V2_BASELINE_V31 == "data/design/policy_v31_v2_baseline.csv"
    assert rnd["holdout"] == L.HOLDOUT_CELL_V31 == PROSPECTIVE_CELL_V31


# --------------------------------------------------------------------------- #
# 8. §9d arm ii — the control, and the flag that makes it expressible
# --------------------------------------------------------------------------- #
def test_arm_ii_drops_the_burnt_columns_to_v3s_exact_51_layout(
        steps: pd.DataFrame) -> None:
    """``--no-burnt`` must land on v3's layout element for element.

    The §9d decomposition only separates "feature" from "cross-fit" and
    "listwise" if arm ii differs from a v3 refit in NOTHING but the corpus it
    reads.  So the assertion is not "51 columns" but "the same 51 columns in the
    same order with the same numbers as ``scalar_features_v3``" -- the v3.1
    vector is sorted, and a drop that merely happened to leave 51 names could
    still have permuted them under the model.  The default path is asserted in
    the same test so the flag cannot buy its behaviour out of the other two.
    """
    from lpopt.policy import train_v3 as T
    from lpopt.policy.v3 import N_SCALARS_V3, N_SCALARS_V31

    x3, n3 = scalar_features_v3(steps)
    arm2, names2, new2 = T.featurize_round(steps, v31=False, no_burnt=True)
    assert names2 == n3 and len(names2) == N_SCALARS_V3 == 51
    np.testing.assert_array_equal(arm2, x3)
    assert not set(NEW_SCALARS_V31) & set(names2)
    assert new2 == list(NEW_SCALARS_V3)

    # arm ii is exactly arm i minus the two columns, not a differently built 51
    x31, names31, _ = T.featurize_round(steps, v31=True)
    assert len(names31) == N_SCALARS_V31 == 53
    keep = [i for i, n in enumerate(names31) if n not in NEW_SCALARS_V31]
    assert [names31[i] for i in keep] == names2
    np.testing.assert_array_equal(x31[:, keep], arm2)

    # the DEFAULT path is untouched by the new keyword
    _, names_dflt, new_dflt = T.featurize_round(steps, v31=False)
    assert (names_dflt, new_dflt) == (n3, list(NEW_SCALARS_V3))


def test_arm_ii_refuses_a_frame_without_the_columns_it_claims_to_drop(
        steps: pd.DataFrame) -> None:
    """Dropping a column that was never there is the v3 round under a new name."""
    from lpopt.policy import train_v3 as T

    bare = steps.drop(columns=list(NEW_SCALARS_V31))
    with pytest.raises(SystemExit, match="does not carry"):
        T.featurize_round(bare, v31=False, no_burnt=True)
    # ... and it may not be combined with the v3.1 SERVING stamp: 51 names
    # under ``policy_move_v31`` is the mis-stamp featurize_round exists to refuse
    with pytest.raises(SystemExit, match="cannot be stamped"):
        T.featurize_round(steps, v31=True, no_burnt=True)


def test_the_arm_ii_flag_stamps_burnt_off_and_refuses_stage_two() -> None:
    from lpopt.policy.train_v3 import assert_v3_path_untouched

    plain = assert_v3_path_untouched(_args())
    assert "burnt" not in plain, "the default v3 stamp gained a key"

    stamp = assert_v3_path_untouched(
        _parser_args(["--no-burnt", "--lam-grid", "0", "--stage2", "off"]))
    assert stamp["burnt"] == "off"
    assert stamp["enabled"] is False and stamp["version"] == "v3"
    assert stamp["crossfit"] is False and stamp["lam_grid"] == [0.0]

    # stage 2 on a 51-scalar vector would stamp policy_move_v31 on it
    with pytest.raises(SystemExit, match="arm ii"):
        assert_v3_path_untouched(
            _parser_args(["--no-burnt", "--stage2", "on", "--xfit-k", "5"]))
    # the lam-grid refusal keeps its teeth: only the registered ZERO is admitted
    with pytest.raises(SystemExit, match="without --stage2 on"):
        assert_v3_path_untouched(_parser_args(["--no-burnt",
                                               "--lam-grid", "0,1"]))
    with pytest.raises(SystemExit, match="without --stage2 on"):
        assert_v3_path_untouched(_parser_args(["--lam-grid", "0"]))


def _parser_args(argv):
    from lpopt.policy.train_v3 import _parser

    return _parser().parse_args(argv)


def test_print_command_renders_the_REGISTERED_v31_values_and_arm_ii(capsys):
    """``--v31 --print-command`` printed the v3 round's defaults in v3.1 flags.

    ``--seeds`` defaulted to 5, ``--base-seed`` to 20260831 and ``--ts`` to
    ``policy_v3``, so the string the launcher offered for copy-paste was a
    command nobody registered -- while the registered launch line in the STEP
    0-b stamp §S0b.10 passes 3 / 20260903 / ``policy_v31`` explicitly.  The two
    have to agree, and arm ii has to be rendered next to arm i now that
    ``--no-burnt`` exists.
    """
    import train_policy_v3 as L

    assert L.main(["--v31", "--print-command"]) == 0
    out = capsys.readouterr().out
    arm_i = next(ln for ln in out.splitlines()
                 if "train_v3" in ln and "--stage2 on" in ln)

    # the registered protocol values of §9b / §9c / §S0b.10
    for token in ("--steps data/policy/steps_v31.parquet",
                  "--cache data/policy/_feature_cache_v31.npz",
                  "--v2-baseline data/design/policy_v31_v2_baseline.csv",
                  f'--holdout-cell "{PROSPECTIVE_CELL_V31}"',
                  "--xfit-k 5",
                  f"--splits {L.SPLITS_V31}",
                  "--stage2 on", "--lam-grid 0,0.3,1.0",
                  "--teacher raw", "--teacher-temp 0.060",
                  "--teacher-eps 0.10",
                  "--stage2-lr 1e-4", "--stage2-epochs 40",
                  "--seeds 3", "--base-seed 20260903",
                  "--epochs 120", "--patience 15",
                  "--batch-size 256", "--lr 1e-3", "--weight-decay 1e-4",
                  "--width 112", "--n-blocks 6", "--protocol revB",
                  "--num-workers 8", "--out-dir runs/policy_v31"):
        assert token in arm_i, token
    assert "runs/policy_v3 " not in arm_i and "steps_v3.parquet" not in arm_i

    arm_ii = next(ln for ln in out.splitlines()
                  if "train_v3" in ln and "--no-burnt" in ln)
    assert "--no-burnt --lam-grid 0 --stage2 off" in arm_ii
    assert "--steps data/policy/steps_v31.parquet" in arm_ii
    assert f'--holdout-cell "{PROSPECTIVE_CELL_V31}"' in arm_ii
    assert "--seeds 3" in arm_ii and "--base-seed 20260903" in arm_ii
    # the emission flags are absent -- with stage 2 off they train nothing
    assert "--xfit-k" not in arm_ii and "--splits" not in arm_ii
    # and it must not write over arm i's run directory
    assert f"--out-dir runs/{L.ARM_II_TS}" in arm_ii
    assert "runs/policy_v31 " not in arm_ii

    # an explicit flag still beats the registered default
    assert L.main(["--v31", "--print-command", "--seeds", "7"]) == 0
    assert "--seeds 7" in capsys.readouterr().out
