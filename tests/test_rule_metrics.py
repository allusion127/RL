"""Engineering-rule metrics RM1-RM6 + their SOFT adoption points.

Three things are pinned here:

1. **Geometry** — the multiplicity-weighted quarter-core counts equal a
   brute-force enumeration on an independently built mirror-expanded 17x17 full
   core.  The oracle in this file does not import ``NEIGH_SLOT``,
   ``PERIPHERY_MASK`` or ``SLOT_WEIGHTS``; it rebuilds them from
   ``ROW_LENGTHS``, so a change to either side breaks the test.
2. **Penalty OFF is byte-identical** — the default ``FlatPowerSpec`` produces
   the exact same ``total`` bits as before the knob existed.
3. **Penalty ON only reorders** — it can move a near-tie, and it is refused for
   every metric the study did NOT validate (RM3/RM4/RM5/RM6).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lpopt.search import acquisition as acq
from lpopt.search import rule_metrics as RM
from lpopt.vendor.masterrl.domain import ROW_LENGTHS, SLOTS, FuelItem, Pattern
from lpopt.vendor.masterrl.ga import _slot_source_coord
from lpopt.vendor.masterrl.surrogate import SurrogatePrediction


# --------------------------------------------------------------------------- #
# independent geometry oracle (no lpopt.data.flatness tables)
# --------------------------------------------------------------------------- #
_CENTER = 8


def _oracle_full_core() -> dict[tuple[int, int], int]:
    """``{(r, c): quarter_slot}`` over the mirror-expanded 17x17 core."""
    cells: dict[tuple[int, int], int] = {}
    idx = 0
    for row, length in enumerate(ROW_LENGTHS):
        for col in range(length):
            for dr in {row, -row}:
                for dc in {col, -col}:
                    cells[(_CENTER + dr, _CENTER + dc)] = idx
            idx += 1
    assert idx == 69 and len(cells) == 241
    return cells


_CELLS = _oracle_full_core()

_FACE = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _oracle_pairs(fresh_slots: set[int], offsets, *, inboard: bool = False) -> int:
    """Brute-force count of UNORDERED full-core pairs, both ends fresh."""
    edge = _oracle_periphery()
    n = 0
    for (r, c), slot in _CELLS.items():
        if slot not in fresh_slots or (inboard and slot in edge):
            continue
        for dr, dc in offsets:
            other = _CELLS.get((r + dr, c + dc))
            if other is None or other not in fresh_slots:
                continue
            if inboard and other in edge:
                continue
            n += 1
    assert n % 2 == 0
    return n // 2


def _oracle_periphery() -> set[int]:
    """Quarter slots with at least one FACE looking out of the fuel region."""
    out = set()
    for (r, c), slot in _CELLS.items():
        if any((r + dr, c + dc) not in _CELLS for dr, dc in _FACE):
            out.add(slot)
    return out


def _oracle_weights() -> np.ndarray:
    w = np.zeros(69)
    for slot in _CELLS.values():
        w[slot] += 1
    return w


def test_oracle_agrees_with_the_library_geometry_tables():
    """The two independent rebuilds must be the same geometry."""
    from lpopt.data.flatness import PERIPHERY_MASK, SLOT_WEIGHTS

    assert np.array_equal(_oracle_weights(), SLOT_WEIGHTS)
    assert _oracle_periphery() == set(np.flatnonzero(PERIPHERY_MASK).tolist())
    assert len(_oracle_periphery()) == 13
    assert _oracle_weights()[sorted(_oracle_periphery())].sum() == 48


# --------------------------------------------------------------------------- #
# hand-built patterns
# --------------------------------------------------------------------------- #
def _pattern(fresh_slots, batch: str = "A") -> Pattern:
    """69-slot pattern with ``fresh_slots`` fresh and the rest self-shuffles."""
    fresh_slots = set(int(s) for s in fresh_slots)
    items = []
    for i in range(len(SLOTS)):
        if i in fresh_slots:
            items.append(FuelItem(kind="fresh", batch=batch))
        else:
            x, y = _slot_source_coord(i)
            items.append(FuelItem(kind="shuffle", restart=1, x=x, y=y))
    return Pattern(tuple(items))


def _chain_pattern(fresh_slots, once_burned: dict[int, int],
                   batch: str = "A") -> Pattern:
    """Fresh slots + slots whose shuffle card points at a FRESH slot (age 2)."""
    items = []
    for i in range(len(SLOTS)):
        if i in fresh_slots:
            items.append(FuelItem(kind="fresh", batch=batch))
        else:
            src = once_burned.get(i, i)
            x, y = _slot_source_coord(src)
            items.append(FuelItem(kind="shuffle", restart=1, x=x, y=y))
    return Pattern(tuple(items))


# --------------------------------------------------------------------------- #
# RM1 / RM2 — hand-checked cases
# --------------------------------------------------------------------------- #
def test_rm1_of_a_single_fresh_assembly_is_zero():
    assert RM.rm_fresh_face_adjacency(_pattern({0})) == 0.0
    assert RM.rm_fresh_diag_adjacency(_pattern({0})) == 0.0


def test_rm1_centre_plus_its_face_neighbour_counts_both_mirror_images():
    """Slot (0,1) has TWO full-core images, both face-touching the centre."""
    slot = next(s.index for s in SLOTS if (s.row, s.col) == (0, 1))
    assert RM.rm_fresh_face_adjacency(_pattern({0, slot})) == 2.0
    assert RM.rm_fresh_diag_adjacency(_pattern({0, slot})) == 0.0


def test_rm2_centre_plus_a_diagonal_neighbour_counts_four_images():
    slot = next(s.index for s in SLOTS if (s.row, s.col) == (1, 1))
    assert RM.rm_fresh_diag_adjacency(_pattern({0, slot})) == 4.0
    assert RM.rm_fresh_face_adjacency(_pattern({0, slot})) == 0.0


def test_rm1_of_a_fully_fresh_core_is_every_face_pair():
    allslots = set(range(69))
    expected = _oracle_pairs(allslots, _FACE)
    assert RM.rm_fresh_face_adjacency(_pattern(allslots)) == float(expected)
    assert RM.rm_fresh_diag_adjacency(_pattern(allslots)) == float(
        _oracle_pairs(allslots, _DIAG))


@pytest.mark.parametrize("seed", range(12))
def test_rm1_rm2_match_a_brute_force_full_core_enumeration(seed):
    rng = np.random.default_rng(seed)
    fresh = set(int(i) for i in rng.choice(69, size=int(rng.integers(1, 40)),
                                           replace=False))
    pat = _pattern(fresh)
    assert RM.rm_fresh_face_adjacency(pat) == float(_oracle_pairs(fresh, _FACE))
    assert RM.rm_fresh_diag_adjacency(pat) == float(_oracle_pairs(fresh, _DIAG))
    assert RM.rm_fresh_face_adjacency(pat, inboard=True) == float(
        _oracle_pairs(fresh, _FACE, inboard=True))
    assert RM.rm_fresh_diag_adjacency(pat, inboard=True) == float(
        _oracle_pairs(fresh, _DIAG, inboard=True))


def test_inboard_variant_never_exceeds_the_whole_core_variant():
    rng = np.random.default_rng(7)
    for _ in range(20):
        fresh = set(int(i) for i in rng.choice(69, size=20, replace=False))
        pat = _pattern(fresh)
        assert (RM.rm_fresh_face_adjacency(pat, inboard=True)
                <= RM.rm_fresh_face_adjacency(pat))


# --------------------------------------------------------------------------- #
# RM4 / RM6
# --------------------------------------------------------------------------- #
def test_rm4_counts_weighted_fresh_on_the_outer_ring_only():
    edge = _oracle_periphery()
    assert RM.rm_fresh_periphery(_pattern(set(range(69)))) == 48.0
    assert RM.rm_fresh_periphery(_pattern({0})) == 0.0      # centre is inboard
    w = _oracle_weights()
    for slot in sorted(edge)[:5]:
        assert RM.rm_fresh_periphery(_pattern({slot})) == float(w[slot])


def test_rm6_is_one_for_an_isolated_fresh_and_zero_for_a_full_core():
    assert RM.rm_checkerboard_degree(_pattern({0})) == 1.0
    assert RM.rm_checkerboard_degree(_pattern(set(range(69)))) == 0.0
    assert np.isnan(RM.rm_checkerboard_degree(_pattern(set())))


def test_rm6_mixes_isolated_and_clustered_fresh_by_weight():
    """Centre (w=1, isolated) + a touching (0,1) pair (w=2, clustered)."""
    s01 = next(s.index for s in SLOTS if (s.row, s.col) == (0, 1))
    far = next(s.index for s in SLOTS if (s.row, s.col) == (5, 5))
    pat = _pattern({0, s01, far})
    w = _oracle_weights()
    # centre and (0,1) touch each other; (5,5) is alone.
    assert RM.rm_checkerboard_degree(pat) == pytest.approx(
        w[far] / (w[0] + w[s01] + w[far]))


# --------------------------------------------------------------------------- #
# RM3
# --------------------------------------------------------------------------- #
def _oracle_mismatch(ri: np.ndarray) -> float:
    """Brute-force ``sum |RI_i - RI_j|`` over full-core unordered face pairs."""
    total = 0.0
    for (r, c), slot in _CELLS.items():
        for dr, dc in _FACE:
            other = _CELLS.get((r + dr, c + dc))
            if other is not None:
                total += abs(float(ri[slot]) - float(ri[other]))
    return total / 2.0


def test_rm3_of_a_uniform_core_is_zero():
    assert RM.rm_reactivity_mismatch(_pattern(set(range(69)))) == 0.0
    # every slot self-rooted -> every RI is age-1 -> still a uniform core.
    assert RM.rm_reactivity_mismatch(_pattern({0})) == 0.0


def test_rm3_matches_a_brute_force_enumeration_of_the_full_core():
    s01 = next(s.index for s in SLOTS if (s.row, s.col) == (0, 1))
    pat = _chain_pattern({0}, {s01: 0})
    ri = RM.reactivity_index(pat)
    assert ri[0] == pytest.approx(1.0)                    # fresh
    assert ri[s01] == pytest.approx(RM.BURN_FACTOR[2])    # once-burned off it
    assert RM.rm_reactivity_mismatch(pat) == pytest.approx(_oracle_mismatch(ri))
    assert RM.rm_reactivity_mismatch(pat) > 0.0


@pytest.mark.parametrize("seed", range(6))
def test_rm3_matches_the_oracle_on_random_chains(seed):
    rng = np.random.default_rng(seed)
    fresh = set(int(i) for i in rng.choice(69, size=17, replace=False))
    burned = {int(i): int(rng.choice(sorted(fresh)))
              for i in range(69) if i not in fresh}
    pat = _chain_pattern(fresh, burned)
    ri = RM.reactivity_index(pat)
    assert RM.rm_reactivity_mismatch(pat) == pytest.approx(_oracle_mismatch(ri))


def test_rm3_is_linear_in_the_reactivity_scale():
    s01 = next(s.index for s in SLOTS if (s.row, s.col) == (0, 1))
    pat = _chain_pattern({0}, {s01: 0}, batch="A")
    base = RM.rm_reactivity_mismatch(pat)
    scaled = RM.rm_reactivity_mismatch(pat, default_enrichment=5.0)
    assert scaled == pytest.approx(5.0 * base)
    # a per-batch enrichment map reaches the fresh slot's own RI.
    assert RM.reactivity_index(pat, {"A": 4.2})[0] == pytest.approx(4.2)


# --------------------------------------------------------------------------- #
# RM5 — synthetic maps
# --------------------------------------------------------------------------- #
def test_rm5_of_a_flat_map_is_the_periphery_weight_fraction():
    assert RM.rm_peripheral_power_share(np.ones(69)) == pytest.approx(48.0 / 241.0)


def test_rm5_rises_when_the_outer_ring_is_hotter():
    from lpopt.data.flatness import PERIPHERY_MASK

    hot = np.ones(69)
    hot[PERIPHERY_MASK] = 2.0
    assert (RM.rm_peripheral_power_share(hot)
            > RM.rm_peripheral_power_share(np.ones(69)))
    cold = np.ones(69)
    cold[PERIPHERY_MASK] = 0.5
    assert (RM.rm_peripheral_power_share(cold)
            < RM.rm_peripheral_power_share(np.ones(69)))


def test_rm5_accepts_the_legacy_4x9x9_stack_and_ignores_nan_slots():
    from lpopt.data.flatness import SLOT_COLS, SLOT_ROWS

    stack = np.full((4, 9, 9), np.nan)
    stack[0][SLOT_ROWS, SLOT_COLS] = 1.0
    assert RM.rm_peripheral_power_share(stack) == pytest.approx(48.0 / 241.0)
    # an unusable map is nan, never an exception.
    assert np.isnan(RM.rm_peripheral_power_share(None))
    assert np.isnan(RM.rm_peripheral_power_share(np.full(69, np.nan)))
    assert np.isnan(RM.rm_peripheral_power_share(np.zeros((3, 3))))


# --------------------------------------------------------------------------- #
# rule_penalty adapter
# --------------------------------------------------------------------------- #
def test_rule_penalty_is_exactly_zero_when_off():
    pats = [_pattern({0, 5}), _pattern(set(range(69)))]
    for weights in (None, {}, {"rm1": 0.0, "rm1i": 0.0}):
        got = RM.rule_penalty(pats, weights)
        assert got.shape == (2,) and np.array_equal(got, np.zeros(2))


def test_rule_penalty_is_the_weighted_sum_of_the_validated_metrics():
    pats = [_pattern(set(range(20)))]
    w = {"rm1": 0.01, "rm2i": 0.5}
    expected = (0.01 * RM.rm_fresh_face_adjacency(pats[0])
                + 0.5 * RM.rm_fresh_diag_adjacency(pats[0], inboard=True))
    assert RM.rule_penalty(pats, w)[0] == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["rm3", "rm4", "rm5", "rm6", "nonsense"])
def test_rule_penalty_refuses_every_unvalidated_metric(bad):
    """Report-only metrics must not be silently usable as shaping terms."""
    with pytest.raises(ValueError, match="not an adopted penalty metric"):
        RM.rule_penalty([_pattern({0})], {bad: 1.0})
    assert bad not in RM.VALIDATED_PENALTY_METRICS


# --------------------------------------------------------------------------- #
# acquisition integration — OFF is byte-identical, ON only reorders
# --------------------------------------------------------------------------- #
def _pred(n: int) -> SurrogatePrediction:
    m = np.zeros((n, 7))
    m[:, 0] = 1.50      # f_r
    m[:, 1] = 1400.0    # cbc
    m[:, 2] = 2.30      # f_q
    m[:, 3] = 625.0     # cyclen
    m[:, 4] = 0.20      # ao
    m[:, 6] = 68.0      # pin bu
    z = np.zeros((n, 7))
    return SurrogatePrediction(m, z.copy(), z.copy())


def _flat_args(n=2):
    return (_pred(n), np.full(n, 1.40), np.zeros(n))


def test_flat_power_default_spec_carries_no_rule_weights():
    assert acq.FlatPowerSpec().rule_penalty_weights is None


def test_flat_power_penalty_off_is_byte_identical():
    spec = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08)
    pats = [_pattern(set(range(20))), _pattern({0})]
    pred, pk, sd = _flat_args()
    base = acq.score_flat_power(pred, pk, sd, spec, np.full(2, 0.30), np.zeros(2))
    # passing patterns with NO weights must not change a single bit...
    same = acq.score_flat_power(pred, pk, sd, spec, np.full(2, 0.30), np.zeros(2),
                                patterns=pats)
    assert base.total.tobytes() == same.total.tobytes()
    assert base.scalar.tobytes() == same.scalar.tobytes()
    assert base.rule_penalty is None and same.rule_penalty is None
    # ...and neither must all-zero weights.
    zero = acq.score_flat_power(
        pred, pk, sd,
        acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08,
                          rule_penalty_weights={"rm1": 0.0}),
        np.full(2, 0.30), np.zeros(2), patterns=pats)
    assert base.total.tobytes() == zero.total.tobytes()


def test_flat_power_penalty_on_reorders_a_flatness_tie():
    """Equal predicted flatness -> the rule breaks the tie toward fewer pairs."""
    clustered = _pattern(set(range(20)))          # many fresh-fresh faces
    spread = _pattern({0})                        # none
    assert (RM.rm_fresh_face_adjacency(clustered)
            > RM.rm_fresh_face_adjacency(spread))
    spec_off = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08)
    spec_on = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08,
                                rule_penalty_weights={"rm1": 0.01})
    pred, pk, sd = _flat_args()
    cov, cs = np.full(2, 0.30), np.zeros(2)
    pats = [clustered, spread]
    off = acq.score_flat_power(pred, pk, sd, spec_off, cov, cs, patterns=pats)
    on = acq.score_flat_power(pred, pk, sd, spec_on, cov, cs, patterns=pats)
    assert off.total[0] == pytest.approx(off.total[1])      # a tie without it
    assert on.total[1] > on.total[0]                        # broken toward spread
    assert on.rule_penalty is not None
    assert on.rule_penalty[0] > on.rule_penalty[1] == 0.0
    # the objective scalar itself is untouched — this is a SOFT term only.
    assert on.scalar.tobytes() == off.scalar.tobytes()


def test_flat_power_penalty_cannot_veto_or_move_the_constraint_tier():
    """A huge weight must never flip feasibility or the F_r safety gate."""
    spec = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08,
                             rule_penalty_weights={"rm1i": 1.0e6})
    pats = [_pattern(set(range(69))), _pattern({0})]
    pred, pk, sd = _flat_args()
    fp = acq.score_flat_power(pred, pk, sd, spec, np.full(2, 0.30), np.zeros(2),
                              patterns=pats)
    assert bool(fp.constraint_ok[0]) and bool(fp.constraint_ok[1])
    assert not fp.fr_gate_violated.any()
    assert fp.constraint_penalty.tolist() == [0.0, 0.0]
    assert np.isfinite(fp.total).all()


def test_flat_power_penalty_needs_patterns_to_apply():
    spec = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08,
                             rule_penalty_weights={"rm1": 0.01})
    pred, pk, sd = _flat_args()
    fp = acq.score_flat_power(pred, pk, sd, spec, np.full(2, 0.30), np.zeros(2))
    assert fp.rule_penalty is None


def test_flat_power_penalty_length_mismatch_is_loud():
    spec = acq.FlatPowerSpec(risk_z=0.0, peak_scale=0.4, cov_scale=0.08,
                             rule_penalty_weights={"rm1": 0.01})
    pred, pk, sd = _flat_args()
    with pytest.raises(ValueError, match="patterns for"):
        acq.score_flat_power(pred, pk, sd, spec, np.full(2, 0.30), np.zeros(2),
                             patterns=[_pattern({0})])


# --------------------------------------------------------------------------- #
# report axis (L-03) — additive, opt-in, never load-bearing
# --------------------------------------------------------------------------- #
def _flat_run_dir(tmp_path, rids):
    import json

    run = tmp_path / "run"
    (run / "waves").mkdir(parents=True, exist_ok=True)
    with open(run / "labels.jsonl", "w", encoding="utf-8") as fh:
        for rid in rids:
            rec = {"record_id": rid, "converged": True, "valid": True,
                   "f_r": 1.50, "cbc_max": 1400.0, "f_q": 2.30, "ao_abs": 0.20,
                   "cyclen": 625.0, "n_cycles": 11.0, "feed": 121,
                   "pattern": "F:K1:0", "node_peak": 1.42, "map_cov": 0.30,
                   # both gated licensing axes MEASURED: the report's best-LP
                   # table is the DELIVERABLE set (review 2026-08-29 §6.4), and
                   # flat_power gates F_xy at 1.65 by default, so an unmeasured
                   # f_xy would leave the table (and this L-03 column) empty.
                   "max_pin_burnup": 68.0, "f_xy": 1.60}
            fh.write(json.dumps({"wave": 0, "slot": "exploit", "origin": "elite",
                                 "record_id": rid, "status": "converged",
                                 "record": rec}) + "\n")
    (run / "status.json").write_text(json.dumps(
        {"status": "complete", "objective": "flat_power", "budget": len(rids),
         "budget_spent": len(rids), "case": "K1_K2-121", "dry_run": True}),
        encoding="utf-8")
    return run


def test_report_l03_column_appears_only_with_a_map_store(tmp_path):
    from lpopt.data.flatness import PERIPHERY_MASK, SLOT_COLS, SLOT_ROWS
    from lpopt.report.report import build_report

    run = _flat_run_dir(tmp_path, ["r0"])
    # no store -> the report is exactly the report it always was.
    plain = build_report(run, pair="K1_K2").read_text(encoding="utf-8")
    assert "periph share" not in plain

    store = tmp_path / "store"
    store.mkdir()
    stack = np.full((4, 9, 9), np.nan, dtype=np.float32)
    plane = np.ones(69)
    plane[PERIPHERY_MASK] = 2.0
    stack[0][SLOT_ROWS, SLOT_COLS] = plane
    np.savez_compressed(store / "maps.npz", r0=stack)

    text = build_report(run, pair="K1_K2", store_dir=store).read_text(encoding="utf-8")
    assert "periph share (L-03)" in text
    assert "REPORT AXIS ONLY" in text
    expected = RM.rm_peripheral_power_share(stack)
    assert f"{expected:.4f}" in text
    # the axis is additive: it changes no verdict, only adds a column.
    assert plain.count("| 1 |") == text.count("| 1 |")


def test_report_l03_column_is_silent_when_the_map_is_missing(tmp_path):
    from lpopt.report.report import build_report, _peripheral_shares

    run = _flat_run_dir(tmp_path, ["r0"])
    store = tmp_path / "empty_store"
    store.mkdir()
    assert _peripheral_shares(store, ["r0"]) == {}
    assert _peripheral_shares(None, ["r0"]) == {}
    assert _peripheral_shares(tmp_path / "nope", ["r0"]) == {}
    assert "periph share" not in build_report(
        run, pair="K1_K2", store_dir=store).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# generator bias + config defaults
# --------------------------------------------------------------------------- #
def test_rule_bias_generator_is_off_by_default():
    from lpopt.config import AcquisitionConfig, ProduceConfig, StratumConfig
    from lpopt.search import produce as P

    acq_cfg = AcquisitionConfig()
    for name in RM.VALIDATED_PENALTY_METRICS:
        assert getattr(acq_cfg, f"flatpower_rule_penalty_{name}") == 0.0
    # the default stratum mix does not name rule_biased at all.
    assert "rule_biased" not in StratumConfig(name="s").generators
    assert ProduceConfig().rule_bias_metric == "rm1"
    assert P._rule_bias_metric(ProduceConfig()) is P._RULE_BIAS_METRICS["rm1"]
    with pytest.raises(ValueError, match="rule_bias_metric"):
        P._rule_bias_metric(ProduceConfig(rule_bias_metric="rm4"))


class _BiasShim:
    """The driver surface ``_rule_biased_genome`` actually touches."""

    def __init__(self, produce_cfg):
        from lpopt.search.produce import ProduceDriver

        self.produce = produce_cfg
        self._rule_bias_cut: dict = {}
        self.logs: list[str] = []
        self._rule_bias_threshold = ProduceDriver._rule_bias_threshold.__get__(self)

    def _log(self, msg):
        self.logs.append(msg)


def test_rule_biased_generator_draws_below_its_own_decile_cut():
    import random as _random

    from lpopt.config import ProduceConfig, StratumConfig
    from lpopt.search.produce import ProduceDriver

    cfg = ProduceConfig(rule_bias_calib=64, rule_bias_tries=24)
    shim = _BiasShim(cfg)
    strat = StratumConfig(name="s", pairs=["K1_K2"], feed=121)
    rng = _random.Random(20260729)
    cut = ProduceDriver._rule_bias_threshold(shim, rng, "K1_K2", 30, strat)
    assert np.isfinite(cut) and cut > 0.0
    assert any("rule_biased generator" in m for m in shim.logs)
    drawn = [ProduceDriver._rule_biased_genome(shim, rng, "K1_K2", 30, strat)
             for _ in range(200)]
    vals = np.array([RM.rm_fresh_face_adjacency(g.to_pattern()) for g in drawn])
    assert all(g.n_fresh == 30 for g in drawn)          # never starves / degrades
    assert np.mean(vals) < cut                          # the bias actually bites
    # ...and it shifts the sampler's distribution DOWN, without walling it off:
    from lpopt.search.genome import random_genome

    plain = np.array([
        RM.rm_fresh_face_adjacency(random_genome(rng, "K1_K2", 30).to_pattern())
        for _ in range(400)])
    assert np.mean(vals) < np.mean(plain)
    # the metric is coarse (a small even integer), so the p90 cut sits on a tie
    # and the realized reject rate is a few percent rather than a clean 10%.
    assert (vals > cut).mean() < 0.02      # only the bounded-retry tail survives
    assert (plain > cut).mean() > 0.02     # which the plain sampler does NOT do


def test_rule_bias_rejection_shifts_the_rm1_distribution_downward():
    """The generator is a BIAS: bounded retries, lower mean, never starving."""
    import random as _random

    rng = _random.Random(20260729)
    pool = [_pattern(set(rng.sample(range(69), 20))) for _ in range(200)]
    values = np.array([RM.rm_fresh_face_adjacency(p) for p in pool])
    cut = float(np.percentile(values, 90.0))
    kept = values[values <= cut]
    assert kept.size >= int(0.85 * values.size)      # a DECILE is rejected, no more
    assert kept.mean() < values.mean()
    assert kept.max() <= cut
