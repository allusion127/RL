"""Campaign A (``intervention_wave``): balance, pairing, schema, resume, analyze.

Everything here runs against a SYNTHETIC store built from
``lpopt.search.genome.random_genome`` and a two-row synthetic fuel table, so the
tests are hermetic and cheap and they exercise BOTH direction regimes the real
frontier contains:

* a **signed** cell, whose two fresh types differ in enrichment, so
  ``batch_swap`` / ``batch_flip`` move the enrichment-weighted radial centre and
  the symmetric-pair draw applies (this is ``T6_T4``/paramA);
* a **degenerate** cell, whose two fresh types share an enrichment, so those two
  classes leave ``fresh_enr_r_center`` EXACTLY unchanged and are drawn as neutral
  arms instead (this is every ga80 pair measured: ``E1``/``E2`` are both 5.000
  w/o, ``N1``/``N2`` both 5.400).

Getting that second case wrong is not a cosmetic failure: it would have
registered 2 of 8 chains per parent on strata that are empty by construction on
three of the five cells, i.e. a quarter of the ga80 budget spent on nothing.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ablation_wave as W                                       # noqa: E402
import intervention_wave as I                                   # noqa: E402
import mine_policy_corpus as M                                  # noqa: E402
from lpopt.data.schema import pack_pattern                      # noqa: E402
from lpopt.search.genome import random_genome                   # noqa: E402

PAIR = "A1_A2"
FEED = 121
LIBRARY = "testlib"

SIGNED = I.Cell("signed", PAIR, FEED, LIBRARY, "paramA", "unequal enrichment")
DEGEN = I.Cell("degen", PAIR, FEED, LIBRARY, "paramA", "iso-enrichment")


# --------------------------------------------------------------------------- #
# synthetic store
# --------------------------------------------------------------------------- #
def _boards(n: int, seed: int = 20260829) -> list[str]:
    rng = random.Random(seed)
    return [pack_pattern(random_genome(rng, PAIR, 30).to_pattern())
            for _ in range(n)]


def _store(tmp_path: Path, n: int = 6, *, feasible: bool = True) -> Path:
    rows = []
    for i, packed in enumerate(_boards(n)):
        rows.append({
            # sha-shaped, and DISTINCT IN ITS FIRST BYTES: an id scheme whose
            # rows share a prefix would hide a pair_id collision.
            "record_id": hashlib.sha256(f"rec{i}".encode()).hexdigest(),
            "case_pair": PAIR, "feed": FEED, "library_id": LIBRARY,
            "pattern": packed, "campaign": "synthetic", "dataset": "P",
            "generator": "synthetic", "parent_record_id": None,
            "converged": True,
            # Feasible cell: F_r under the 1.55 limit.  Infeasible cell: over it,
            # which is the real HGD569 f125 situation (best F_r 1.6036).
            "f_r": (1.45 + 0.01 * i) if feasible else (1.60 + 0.01 * i),
            "f_q": 2.0, "cbc_max": 1200.0, "ao_abs": 0.05,
            "cyclen": 500.0 + i, "node_peak": 1.30 + 0.001 * i,
            "map_cov": 0.2, "f_xy": 1.60 + 0.001 * i, "f_xya": 1.5,
            "e_core": 4.8, "restart_provenance": "native:MAS_RST.TEST",
        })
    path = tmp_path / "records.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _fuel(tmp_path: Path, *, iso: bool) -> Path:
    path = tmp_path / "fuel_types.parquet"
    pd.DataFrame([
        {"library_id": LIBRARY, "type_id": "A1", "u_avg_enrichment": 4.5},
        {"library_id": LIBRARY, "type_id": "A2",
         "u_avg_enrichment": 4.5 if iso else 5.0},
    ]).to_parquet(path, index=False)
    return path


def _plan(tmp_path: Path, cell: I.Cell, *, iso: bool, parents: int = 3,
          feasible: bool = True, monkeypatch=None) -> dict:
    out = tmp_path / f"plan_{cell.name}.json"
    if monkeypatch is not None:
        monkeypatch.setattr(I, "CELLS_BY_NAME", {cell.name: cell})
        monkeypatch.setattr(I, "CELLS_R1", (cell,))
    args = SimpleNamespace(
        cells=[cell.name], kit=None, out=str(out),
        store=str(_store(tmp_path, feasible=feasible)),
        fuel_types=str(_fuel(tmp_path, iso=iso)),
        parents=parents, seed=I.SEED, wave="test")
    assert I.cmd_plan(args) == 0
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# balance + pairing
# --------------------------------------------------------------------------- #
def test_signed_cell_draws_symmetric_dose_matched_pairs(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, monkeypatch=monkeypatch)
    cand = pd.DataFrame(man["candidates"])
    paid = cand[cand["source"] == "paid"]

    assert man["cells"][0]["direction_regime"]["batch_swap"] == "signed"
    assert man["cells"][0]["shortfalls"] == []
    # 8 chains per parent, every parent.
    per_parent = paid.groupby("parent_record_id").size()
    assert set(per_parent) == {I.MOVES_PER_PARENT}

    for prid, blk in paid.groupby("parent_record_id"):
        mix = blk.groupby(["move_class", "fresh_radial_dir"]).size().to_dict()
        assert mix[("fresh_relocate", "outward")] == 2
        assert mix[("fresh_relocate", "inward")] == 2
        assert mix[("batch_swap", "outward")] == 1
        assert mix[("batch_swap", "inward")] == 1
        assert mix[("rewire_swap", "neutral")] == 1
        assert sum(v for (c, _), v in mix.items() if c == "batch_flip") == 1

        # Every paired row belongs to a two-member, one-outward-one-inward,
        # same-parent, same-class, same-burn-state pair.
        pairs = blk[blk["pair_id"].notna()]
        assert len(pairs) == 6
        for pid, pair in pairs.groupby("pair_id"):
            assert len(pair) == 2
            assert set(pair["pair_role"]) == {"outward", "inward"}
            assert pair["move_class"].nunique() == 1
            assert pair["burn_state"].nunique() == 1
            assert set(pair["parent_record_id"]) == {prid}


def test_degenerate_cell_falls_back_to_neutral_not_shortfall(tmp_path, monkeypatch):
    man = _plan(tmp_path, DEGEN, iso=True, monkeypatch=monkeypatch)
    cand = pd.DataFrame(man["candidates"])
    paid = cand[cand["source"] == "paid"]
    regime = man["cells"][0]["direction_regime"]

    assert regime["batch_swap"] == "degenerate"
    assert regime["batch_flip"] == "degenerate"
    assert regime["fresh_relocate"] == "signed"
    # The budget is NOT lost: the class is drawn as a neutral arm of the same
    # size, and nothing is reported as missing.
    assert man["cells"][0]["shortfalls"] == []
    assert set(paid.groupby("parent_record_id").size()) == {I.MOVES_PER_PARENT}

    for _prid, blk in paid.groupby("parent_record_id"):
        mix = blk.groupby(["move_class", "fresh_radial_dir"]).size().to_dict()
        assert mix[("batch_swap", "neutral")] == 2
        assert mix[("batch_flip", "neutral")] == 1
        assert mix[("fresh_relocate", "outward")] == 2
        assert mix[("fresh_relocate", "inward")] == 2
        assert ("batch_swap", "outward") not in mix
    degen = paid[paid["pair_role"] == "neutral_degenerate"]
    assert set(degen["move_class"]) == {"batch_swap", "batch_flip"}
    # Only fresh_relocate can still be paired here.
    assert set(paid[paid["pair_id"].notna()]["move_class"]) == {"fresh_relocate"}


def test_burn_state_is_spread_and_classified_from_parent_depths(tmp_path,
                                                                monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, monkeypatch=monkeypatch)
    cand = pd.DataFrame(man["candidates"])
    paid = cand[cand["source"] == "paid"]
    assert set(paid["burn_state"]) <= set(I.BURN_STATES)
    # fresh_relocate touches a burned unit by definition, so it can never be
    # classified "fresh"; batch_flip/batch_swap only ever touch fresh units.
    assert set(paid[paid["move_class"] == "fresh_relocate"]["burn_state"]) <= {
        "once", "twice_plus"}
    assert set(paid[paid["move_class"].isin(["batch_flip", "batch_swap"])]
               ["burn_state"]) <= {"fresh", "center"}


def test_burn_state_class_matches_genome_depths() -> None:
    packed = _boards(1)[0]
    g = M.genome_of(packed)
    depths = g._depths()
    for _cls, child, _tag in W.enumerate_single_moves(g)[:60]:
        cpacked = pack_pattern(child.to_pattern())
        units = I.changed_units(packed, cpacked)
        got = I.burn_state_class(g, packed, cpacked)
        if not units:
            assert got == "center"
            continue
        deepest = max(int(depths.get(u, 0)) for u in units)
        assert got == ("fresh" if deepest == 0
                       else "once" if deepest == 1 else "twice_plus")


def test_pick_burnstate_balanced_round_robins_then_truncates() -> None:
    frame = pd.DataFrame({
        "burn_state": ["fresh"] * 5 + ["once"] * 5 + ["twice_plus"] * 1,
        "dose": np.linspace(0.0, 1.0, 11),
        "move_tag": [f"t{i:02d}" for i in range(11)],
    })
    idx = I.pick_burnstate_balanced(frame, 3, "dose")
    assert len(idx) == 3
    assert set(frame.loc[idx, "burn_state"]) == {"fresh", "once", "twice_plus"}
    # Deterministic and stable.
    assert idx == I.pick_burnstate_balanced(frame, 3, "dose")
    # Asking for more than exists returns everything, never raises.
    assert len(I.pick_burnstate_balanced(frame, 99, "dose")) == 11


# --------------------------------------------------------------------------- #
# manifest schema
# --------------------------------------------------------------------------- #
def test_manifest_schema(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, monkeypatch=monkeypatch)
    for key in ("written_utc", "wave", "generator", "seed", "deck_knobs",
                "moves_per_parent", "quota", "kits", "cells", "candidates",
                "n_paid", "n_free", "budget_cap_per_cell", "burn_states"):
        assert key in man, key
    cell = man["cells"][0]
    for key in ("name", "cell", "pair", "feed", "library_id", "kit", "campaign",
                "lineage_source", "run_subdir", "direction_regime", "parents",
                "parent_hamming", "shortfalls", "neighbourhood_census",
                "parent_restart_provenance", "n_feasible_parents"):
        assert key in cell, key
    assert cell["kit"] == "paramA"
    assert cell["campaign"] == "intervention_signed"
    assert cell["lineage_source"] == "intervention_signed"

    cand = pd.DataFrame(man["candidates"])
    for col in ("cell", "record_id", "parent_record_id", "pattern", "move_class",
                "fresh_radial_dir", "burn_state", "dose", "pair_id", "pair_role",
                "stratum", "source", "campaign", "case_pair", "feed",
                "library_id", "parent_f_r", "parent_f_xy", "parent_node_peak"):
        assert col in cand.columns, col
    assert set(cand["cell"]) == {"signed"}
    assert man["n_paid"] == int((cand["source"] == "paid").sum())
    # record_id is the canonical store identity, so a planned child and a
    # merged row are the same row.
    from lpopt.data.schema import compute_record_id
    row = cand.iloc[0]
    assert row["record_id"] == compute_record_id(
        row["pattern"], LIBRARY, PAIR, man["deck_knobs"])
    # Every paid child is a genuinely NEW board (dedup against the store).
    assert cand[cand["source"] == "paid"]["record_id"].is_unique


def test_plan_seats_parents_on_an_infeasible_cell(tmp_path, monkeypatch):
    """The HGD569 case: no board in the cell passes the program limits."""
    man = _plan(tmp_path, SIGNED, iso=False, parents=3, feasible=False,
                monkeypatch=monkeypatch)
    cell = man["cells"][0]
    assert cell["n_parents"] == 3
    assert cell["n_feasible_parents"] == 0
    assert all(p["family"].endswith("_infeasible") or p["family"] == "topup"
               for p in cell["parents"])
    assert man["n_paid"] == 3 * I.MOVES_PER_PARENT


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
def test_resume_reruns_harness_failures_only(tmp_path, monkeypatch):
    """A per-cell resume must pick harness faults back up, keep physics answers.

    Same contract as ``tests/test_ablation_resume.py``, asserted at the level the
    multi-cell runner actually uses it: one results jsonl per CELL run dir.
    """
    man = _plan(tmp_path, SIGNED, iso=False, monkeypatch=monkeypatch)
    paid = [c["record_id"] for c in man["candidates"] if c["source"] == "paid"]
    cell_dir = tmp_path / "runs" / SIGNED.run_subdir
    cell_dir.mkdir(parents=True)
    rows = [
        {"record_id": paid[0], "status": "converged", "failure": ""},
        {"record_id": paid[1], "status": "error", "failure": "non_finite_flux"},
        {"record_id": paid[2], "status": "error",
         "failure": "[Errno 28] No space left on device"},
        {"record_id": paid[3], "status": "nonconverged", "failure": ""},
    ]
    (cell_dir / I.RESULTS_NAME).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    done = W._done(cell_dir / I.RESULTS_NAME)
    assert done == {paid[0], paid[1], paid[3]}
    todo = [r for r in paid if r not in done]
    assert paid[2] in todo                 # the disk failure is re-run
    assert len(todo) == len(paid) - 3

    # The sub-plan the runner hands to ablation_wave carries exactly this cell.
    sub = I.write_subplan(man, man["cells"][0], cell_dir / I.SUBPLAN_NAME)
    loaded = json.loads(sub.read_text(encoding="utf-8"))
    assert {c["cell"] for c in loaded["candidates"]} == {"signed"}
    assert len([c for c in loaded["candidates"]
                if c["source"] == "paid"]) == len(paid)


def test_run_refuses_to_mix_libraries(tmp_path, monkeypatch):
    """Asset routing is per-library; one --package cannot serve two of them."""
    man = _plan(tmp_path, SIGNED, iso=False, monkeypatch=monkeypatch)
    man["cells"].append(dict(man["cells"][0], name="other", kit="ga80",
                             library_id="ga80"))
    plan_path = tmp_path / "mixed.json"
    plan_path.write_text(json.dumps(man), encoding="utf-8")
    args = SimpleNamespace(plan=str(plan_path), cells=None, kit=None,
                           package=str(tmp_path), fuel_types="x", exe="x",
                           run_dir=str(tmp_path / "runs"), workers=1,
                           host_reserve=1, wave_size=1, max_cycles=1,
                           timeout=1.0, max_chains=0, allow_fallback=False,
                           allow_restart_drift=False, dry_run=True)
    with pytest.raises(SystemExit, match="PER LIBRARY"):
        I.cmd_run(args)


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _label(man: dict, run_dir: Path, *, effect: float,
           offset: float = 0.0) -> None:
    """Write synthetic MASTER results + an F_xy sidecar with a KNOWN effect.

    Every outward child gets ``d_cyclen = -effect`` and every inward child
    ``+effect`` relative to its parent, so the paired contrast must recover
    ``-2*effect`` with a 0/n sign test.

    ``offset`` adds a COMMON-MODE shift to every child of the cell, direction
    and class alike - the HGD569_f125 baseline shift of s7, in miniature.  It
    biases every vs-parent table and cancels in the paired contrast.
    """
    cell = man["cells"][0]
    cdir = run_dir / cell["run_subdir"]
    cdir.mkdir(parents=True, exist_ok=True)
    parents = {p["record_id"]: p for p in cell["parents"]}
    res, side = [], []
    for c in man["candidates"]:
        if c["source"] != "paid":
            continue
        p = parents[c["parent_record_id"]]
        sign = {"outward": -1.0, "inward": +1.0}.get(c["fresh_radial_dir"], 0.0)
        res.append({
            "record_id": c["record_id"],
            "parent_record_id": c["parent_record_id"],
            "move_class": c["move_class"],
            "fresh_radial_dir": c["fresh_radial_dir"],
            "status": "converged", "failure": "",
            "node_peak": p["node_peak"] + 0.01 * sign + offset,
            "fom": {"F_r": p["f_r"] + 0.02 * sign + offset,
                    "cyclen": p["cyclen"] + effect * sign + offset,
                    "CBC_max": p["cbc_max"], "F_q": p["f_q"]},
        })
        side.append({"record_id": c["record_id"], "status": "converged",
                     "f_xy": p["f_xy"] + 0.005 * sign + offset, "f_xya": 1.5,
                     "fxy_sane": True, "fxy_reason": ""})
    (cdir / I.RESULTS_NAME).write_text(
        "\n".join(json.dumps(r) for r in res) + "\n", encoding="utf-8")
    (cdir / I.FXY_SIDECAR_NAME).write_text(
        "\n".join(json.dumps(r) for r in side) + "\n", encoding="utf-8")


def test_analyze_recovers_the_injected_paired_effect(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, parents=5, monkeypatch=monkeypatch)
    run_dir = tmp_path / "runs"
    _label(man, run_dir, effect=1.5)

    frame = I.wave_frame(man, run_dir)
    assert len(frame) == man["n_paid"]
    assert frame["converged"].all()
    # F_xy comes from the sidecar, i.e. from MAS_OUT via lpopt.data.fxy.
    assert frame["child_f_xy"].notna().all()

    pooled = I.effect_table(frame, ["move_class", "fresh_radial_dir"])
    out = pooled[(pooled["move_class"] == "fresh_relocate")
                 & (pooled["fresh_radial_dir"] == "outward")].iloc[0]
    inw = pooled[(pooled["move_class"] == "fresh_relocate")
                 & (pooled["fresh_radial_dir"] == "inward")].iloc[0]
    assert out["mean_cyclen"] == pytest.approx(-1.5)
    assert inw["mean_cyclen"] == pytest.approx(+1.5)
    assert out["mean_F_xy"] == pytest.approx(-0.005)
    assert out["improving_F_r"] == pytest.approx(1.0)

    pairs = I.paired_contrasts(frame, ["move_class"])
    fr = pairs[(pairs["move_class"] == "fresh_relocate")
               & (pairs["response"] == "cyclen")].iloc[0]
    assert fr["n_pairs"] == 2 * 5                    # 2 pairs x 5 parents
    assert fr["mean_out_minus_in"] == pytest.approx(-3.0)
    assert fr["sign_pos"] == 0                       # every pair negative
    assert fr["sign_p"] < 0.01
    # rewire_swap is the neutral control: no pairs, so no paired row.
    assert "rewire_swap" not in set(pairs["move_class"])

    signs = I.parent_blocked_signs(frame, ["cell", "move_class",
                                           "fresh_radial_dir"])
    row = signs[(signs["move_class"] == "fresh_relocate")
                & (signs["fresh_radial_dir"] == "outward")
                & (signs["response"] == "cyclen")].iloc[0]
    assert row["n_parents"] == 5                     # the analysis unit
    assert row["sign_pos"] == 0 and row["sign_n"] == 5


def test_analyze_ignores_unconverged_and_missing_fxy(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, parents=3, monkeypatch=monkeypatch)
    run_dir = tmp_path / "runs"
    _label(man, run_dir, effect=1.0)
    cdir = run_dir / man["cells"][0]["run_subdir"]
    # Flip one chain to a physics kill and drop its sidecar row.
    lines = (cdir / I.RESULTS_NAME).read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first.update({"status": "error", "failure": "non_finite_flux", "fom": None,
                  "node_peak": None})
    lines[0] = json.dumps(first)
    (cdir / I.RESULTS_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    side = [json.loads(x) for x in
            (cdir / I.FXY_SIDECAR_NAME).read_text(encoding="utf-8").splitlines()]
    (cdir / I.FXY_SIDECAR_NAME).write_text(
        "\n".join(json.dumps(s) for s in side[1:]) + "\n", encoding="utf-8")

    frame = I.wave_frame(man, run_dir)
    killed = frame[frame["record_id"] == first["record_id"]].iloc[0]
    assert not killed["converged"]
    assert np.isnan(killed["d_cyclen"]) and np.isnan(killed["d_f_xy"])
    # ...and it is excluded from every table rather than counted as a zero.
    pooled = I.effect_table(frame, ["move_class", "fresh_radial_dir"])
    assert int(pooled["n"].sum()) == man["n_paid"] - 1


def test_fxy_sidecar_roundtrip(tmp_path) -> None:
    path = tmp_path / I.FXY_SIDECAR_NAME
    path.write_text(
        json.dumps({"record_id": "a", "f_xy": 1.7}) + "\n"
        "not json\n\n"
        + json.dumps({"record_id": "a", "f_xy": 1.8}) + "\n"
        + json.dumps({"record_id": "b", "f_xy": None}) + "\n",
        encoding="utf-8")
    table = I.load_fxy_sidecar(path)
    assert table["a"]["f_xy"] == 1.8          # a rerun supersedes
    assert table["b"]["f_xy"] is None
    assert I.load_fxy_sidecar(tmp_path / "nope.jsonl") == {}


def test_fxy_row_reads_the_verifier_outcome() -> None:
    """``_fxy_row`` reads WaveOutcome.fxy - the parsed MAS_OUT, not a proxy."""
    from lpopt.data.fxy import FxyResult

    peaks = FxyResult(f_xy=1.71, f_xya=1.55, steps=(), n_steps=4, sane=True,
                      reason="")
    oc = SimpleNamespace(meta={"record_id": "abc"}, status="converged", fxy=peaks)
    row = I._fxy_row(oc)
    assert row == {"record_id": "abc", "status": "converged", "f_xy": 1.71,
                   "f_xya": 1.55, "fxy_n_steps": 4, "fxy_sane": True,
                   "fxy_reason": "", "fxy_efpd_max": None}
    # A purged / physics-killed dir yields fxy=None and must not raise.
    none_row = I._fxy_row(SimpleNamespace(meta={"record_id": "d"},
                                          status="error", fxy=None))
    assert none_row["f_xy"] is None and none_row["fxy_reason"] == "no_result"


# --------------------------------------------------------------------------- #
# corpus append
# --------------------------------------------------------------------------- #
STEP_COLS = ["lineage_source", "campaign", "parent_record_id", "child_record_id",
             "move_class", "fresh_radial_dir", "single_move", "d_f_r"]


def _steps(tmp_path: Path) -> Path:
    path = tmp_path / "steps.parquet"
    pd.DataFrame([{
        "lineage_source": "sa_mocha", "campaign": "old",
        "parent_record_id": "p0", "child_record_id": "c0",
        "move_class": "batch_swap", "fresh_radial_dir": "inward",
        "single_move": True, "d_f_r": 0.1,
    }])[STEP_COLS].to_parquet(path, index=False)
    return path


def _corpus_args(tmp_path: Path, plan: Path, steps: Path, **kw):
    return SimpleNamespace(plan=str(plan), cells=None, kit=None,
                           store=str(tmp_path / "records.parquet"),
                           steps=str(steps), wave="test", dry_run=False, **kw)


def test_corpus_append_is_idempotent(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, parents=2, monkeypatch=monkeypatch)
    plan_path = tmp_path / "plan_signed.json"
    steps = _steps(tmp_path)

    mined = pd.DataFrame([{
        "lineage_source": SIGNED.lineage_source, "campaign": SIGNED.campaign,
        "parent_record_id": "p1", "child_record_id": "c1",
        "move_class": "fresh_relocate", "fresh_radial_dir": "outward",
        "single_move": True, "d_f_r": -0.02,
    }])[STEP_COLS]

    import ablation_analyze as A
    monkeypatch.setattr(A, "build_wave_steps",
                        lambda campaign, lineage: mined.assign(
                            lineage_source=lineage))

    args = _corpus_args(tmp_path, plan_path, steps)
    assert I.cmd_corpus(args) == 0
    after = pd.read_parquet(steps)
    assert len(after) == 2
    assert set(after["lineage_source"]) == {"sa_mocha", "intervention_signed"}

    # Re-running must not duplicate the edge: dedup is on (parent, child).
    assert I.cmd_corpus(_corpus_args(tmp_path, plan_path, steps)) == 0
    assert len(pd.read_parquet(steps)) == 2
    # The pre-append backup exists and still holds the ORIGINAL corpus.
    backup = steps.with_suffix(".parquet.bak_pre_test")
    assert backup.exists() and len(pd.read_parquet(backup)) == 1


def test_corpus_refuses_schema_drift(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, parents=2, monkeypatch=monkeypatch)
    plan_path = tmp_path / "plan_signed.json"
    steps = _steps(tmp_path)
    assert man["cells"][0]["lineage_source"] == "intervention_signed"

    import ablation_analyze as A
    monkeypatch.setattr(A, "build_wave_steps", lambda campaign, lineage:
                        pd.DataFrame([{"lineage_source": lineage,
                                       "parent_record_id": "p1",
                                       "child_record_id": "c1"}]))
    with pytest.raises(SystemExit, match="schema drift"):
        I.cmd_corpus(_corpus_args(tmp_path, plan_path, steps))
    assert len(pd.read_parquet(steps)) == 1        # untouched


def test_corpus_dry_run_writes_nothing(tmp_path, monkeypatch):
    _plan(tmp_path, SIGNED, iso=False, parents=2, monkeypatch=monkeypatch)
    steps = _steps(tmp_path)
    import ablation_analyze as A
    monkeypatch.setattr(A, "build_wave_steps", lambda campaign, lineage:
                        pd.DataFrame([{
                            "lineage_source": lineage, "campaign": campaign,
                            "parent_record_id": "p1", "child_record_id": "c1",
                            "move_class": "batch_flip",
                            "fresh_radial_dir": "outward",
                            "single_move": True, "d_f_r": 0.0}])[STEP_COLS])
    args = _corpus_args(tmp_path, tmp_path / "plan_signed.json", steps)
    args.dry_run = True
    assert I.cmd_corpus(args) == 0
    assert len(pd.read_parquet(steps)) == 1


# --------------------------------------------------------------------------- #
# registered constants
# --------------------------------------------------------------------------- #
def test_registered_quota_sums_to_the_registered_budget() -> None:
    assert I.QUOTA_TOTAL == I.MOVES_PER_PARENT == 8
    assert I.N_PARENTS * I.MOVES_PER_PARENT * len(I.CELLS_R1) == 800
    assert len({c.name for c in I.CELLS_R1}) == len(I.CELLS_R1)
    # Two libraries -> two kits, and the runner must be invoked once per kit.
    assert {c.kit for c in I.CELLS_R1} == {"paramA", "ga80"}
    assert I.BUDGET_CAP_PER_CELL >= I.N_PARENTS * I.MOVES_PER_PARENT


def test_resolve_cells_rejects_unknown_names() -> None:
    assert len(I.resolve_cells(None)) == len(I.CELLS_R1)
    assert [c.name for c in I.resolve_cells(["E1E2_f109"])] == ["E1E2_f109"]
    with pytest.raises(SystemExit, match="unknown cell"):
        I.resolve_cells(["nope"])


# --------------------------------------------------------------------------- #
# slot-geometry direction degeneracy (results 20260830 s5.1 / s11-1)
# --------------------------------------------------------------------------- #
def _three_type_boards() -> tuple[str, str, str, str]:
    """A 3-fresh-type parent, an outward child, its CORE TWIN, and a distinct one.

    The twin is the outward child's diagonal transpose: a different 69-slot
    string, a different ``record_id``, the SAME reactor.  It is the cheapest
    exact instance of the degeneracy round 1 paid 40 chains for - there the two
    siblings differed in five slots and still produced a bit-identical power map
    on all 20 parents.
    """
    from lpopt.data.geometry import transpose
    from lpopt.data.schema import unpack_pattern

    base = _boards(1, seed=7)[0].split("|")
    fresh = [i for i, t in enumerate(base) if t.startswith("F:")]
    assert len(fresh) >= 6
    parent = list(base)
    for i in fresh[:2]:
        parent[i] = "F:A3:0"                      # a genuine THIRD fresh type
    packed = "|".join(parent)

    def relabel(idx: int, batch: str) -> str:
        toks = list(parent)
        toks[idx] = f"F:{batch}:0"
        return "|".join(toks)

    out = relabel(fresh[2], "A3")
    other = relabel(fresh[3], "A3")
    twin = pack_pattern(transpose(unpack_pattern(out)))
    assert len({packed, out, other, twin}) == 4
    return packed, out, twin, other


def test_core_digest_sees_through_the_encoding_but_not_through_physics() -> None:
    packed, out, twin, other = _three_type_boards()
    assert I.core_digest(out) == I.core_digest(twin)      # one core, two strings
    assert I.core_digest(out) != I.core_digest(other)     # two cores
    assert I.core_digest(out) != I.core_digest(packed)
    # ...and it is NOT the plain encoding digest, which the dedup already has.
    from lpopt.data.schema import unpack_pattern
    assert unpack_pattern(out).digest != unpack_pattern(twin).digest


def _pair_frame(children: list[tuple[str, str, float]]):
    """``draw_parent_sample``-shaped frame: (pattern, direction, dose) rows."""
    return pd.DataFrame([{
        "pattern": p, "move_class": "batch_swap", "fresh_radial_dir": d,
        "dose": dose, "burn_state": "fresh", "move_tag": f"bs:{i}",
        "swap_span": 1.0, "n_slots_changed": 2,
    } for i, (p, d, dose) in enumerate(children)])


def test_pairing_refuses_a_sibling_that_realizes_the_outward_childs_core() -> None:
    _packed, out, twin, other = _three_type_boards()
    # The twin is the PERFECT dose match, so the old rule would take it; the
    # honest sibling is one dose-step away.
    frame = _pair_frame([(out, "outward", 0.10),
                         (twin, "inward", 0.10),
                         (other, "inward", 0.11)])
    picked, meta, short = I.draw_parent_sample(frame, "parent0",
                                               random.Random(0))
    pairs = {meta[i]["pair_role"]: frame.at[i, "pattern"] for i in picked
             if meta[i]["pair_id"] is not None}
    assert pairs["outward"] == out
    assert pairs["inward"] == other, "the core twin must not be bought"
    assert I.core_digest(pairs["outward"]) != I.core_digest(pairs["inward"])
    assert not [s for s in short if s["move_class"] == "batch_swap"]


def test_a_wholly_degenerate_class_is_a_named_shortfall_not_a_silent_pair() -> None:
    _packed, out, twin, _other = _three_type_boards()
    frame = _pair_frame([(out, "outward", 0.10), (twin, "inward", 0.10)])
    picked, meta, short = I.draw_parent_sample(frame, "parent0",
                                               random.Random(0))
    assert not [i for i in picked if meta[i]["pair_id"] is not None]
    bad = [s for s in short if s["move_class"] == "batch_swap"]
    assert len(bad) == 1
    assert bad[0]["kind"] == "core_degenerate"
    assert bad[0]["got"] == 0 and bad[0]["n_core_degenerate"] == 1


def test_the_guard_does_not_move_the_draw_when_no_core_repeats() -> None:
    """No degeneracy -> the registered ``k``-draw and its pairing, unchanged."""
    _packed, out, _twin, other = _three_type_boards()
    frame = _pair_frame([(out, "outward", 0.10), (other, "inward", 0.11)])
    picked, meta, _short = I.draw_parent_sample(frame, "parent0",
                                                random.Random(0))
    with_guard = [(i, meta[i]["pair_role"]) for i in picked]

    original = I._sibling

    def unguarded(pool, used, dose, state, forbid_core=None):
        return original(pool, used, dose, state, None)

    try:
        I._sibling = unguarded                       # type: ignore[assignment]
        picked2, meta2, _ = I.draw_parent_sample(frame, "parent0",
                                                 random.Random(0))
    finally:
        I._sibling = original                        # type: ignore[assignment]
    assert with_guard == [(i, meta2[i]["pair_role"]) for i in picked2]


def test_plan_hard_fails_if_a_core_degenerate_pair_ever_reaches_the_manifest(
        tmp_path, monkeypatch):
    """Defence in depth: the block-level check, with the draw guard disabled."""
    real = I._sibling
    monkeypatch.setattr(
        I, "_sibling",
        lambda pool, used, dose, state, forbid_core=None: real(
            pool, used, dose, state, None))
    monkeypatch.setattr(I, "core_digest", lambda packed: "ONECORE")
    with pytest.raises(SystemExit, match="core-degenerate"):
        _plan(tmp_path, SIGNED, iso=False, parents=2, monkeypatch=monkeypatch)


def test_plan_reports_the_core_degeneracy_census(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, parents=2, monkeypatch=monkeypatch)
    cell = man["cells"][0]
    assert cell["n_core_degenerate_rejected"] == 0
    assert cell["n_core_degenerate_shortfalls"] == 0
    cand = pd.DataFrame(man["candidates"])
    assert "core_digest" in cand.columns
    # Every drawn pair is two DIFFERENT cores - the property the wave assumed.
    paired = cand[cand["pair_id"].notna()]
    assert (paired.groupby("pair_id")["core_digest"].nunique() == 2).all()


# --------------------------------------------------------------------------- #
# cell-baseline diagnostic (results 20260830 s7 / s11-2)
# --------------------------------------------------------------------------- #
def test_neutral_control_offset_is_measured_and_removed(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, parents=5, monkeypatch=monkeypatch)
    run_dir = tmp_path / "runs"
    _label(man, run_dir, effect=1.5, offset=7.0)
    frame = I.wave_frame(man, run_dir)

    offsets = I.neutral_control_offset(frame, ["cell"])
    row = offsets[offsets["response"] == "cyclen"].iloc[0]
    assert row["cell"] == "signed"
    assert row["n"] == 5 and row["n_parents"] == 5   # one rewire_swap per parent
    assert row["offset"] == pytest.approx(7.0)
    assert offsets[offsets["response"] == "F_xy"]["offset"].iloc[0] \
        == pytest.approx(7.0)

    # RAW is contaminated by the common mode...
    pooled = I.effect_table(frame, ["move_class", "fresh_radial_dir"])
    raw = pooled[(pooled["move_class"] == "fresh_relocate")
                 & (pooled["fresh_radial_dir"] == "outward")].iloc[0]
    assert raw["mean_cyclen"] == pytest.approx(7.0 - 1.5)

    # ...and the adjusted table is the offset-free answer.
    adj = I.adjust_by_neutral_control(frame, offsets, ["cell"])
    pooled_adj = I.effect_table(adj, ["move_class", "fresh_radial_dir"])
    fixed = pooled_adj[(pooled_adj["move_class"] == "fresh_relocate")
                       & (pooled_adj["fresh_radial_dir"] == "outward")].iloc[0]
    assert fixed["mean_cyclen"] == pytest.approx(-1.5)
    assert fixed["mean_F_xy"] == pytest.approx(-0.005)
    # The control arm itself is zero after its own correction, by construction.
    ctrl = pooled_adj[pooled_adj["move_class"] == I.NEUTRAL_CONTROL_CLASS].iloc[0]
    assert ctrl["mean_cyclen"] == pytest.approx(0.0, abs=1e-9)
    # The RAW frame is untouched - both tables are published.
    assert frame["d_cyclen"].equals(I.wave_frame(man, run_dir)["d_cyclen"])
    assert adj["d_cyclen_offset"].dropna().unique().tolist() == [7.0]


def test_paired_contrast_is_immune_to_the_cell_baseline(tmp_path, monkeypatch):
    """A common-mode shift cancels in out-minus-in; the *_adj twin is redundant."""
    man = _plan(tmp_path, SIGNED, iso=False, parents=3, monkeypatch=monkeypatch)
    plain, shifted = tmp_path / "a", tmp_path / "b"
    _label(man, plain, effect=1.5)
    _label(man, shifted, effect=1.5, offset=7.0)
    a = I.paired_contrasts(I.wave_frame(man, plain), ["move_class"])
    b = I.paired_contrasts(I.wave_frame(man, shifted), ["move_class"])
    key = ["move_class", "response"]
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    assert np.allclose(a["mean_out_minus_in"], b["mean_out_minus_in"])
    assert a["sign_pos"].tolist() == b["sign_pos"].tolist()


def test_analyze_writes_the_offset_and_adjusted_tables(tmp_path, monkeypatch):
    man = _plan(tmp_path, SIGNED, iso=False, parents=3, monkeypatch=monkeypatch)
    plan_path = tmp_path / "plan_signed.json"
    run_dir = tmp_path / "runs"
    _label(man, run_dir, effect=1.0, offset=2.0)
    out_dir = tmp_path / "reports"
    assert I.cmd_analyze(SimpleNamespace(
        plan=str(plan_path), cells=None, kit=None, run_dir=str(run_dir),
        out_dir=str(out_dir), wave="test")) == 0
    for name in ("effects_by_cell", "effects_pooled", "effects_by_burn_state",
                 "neutral_control_offset", "effects_by_cell_adj",
                 "effects_pooled_adj", "effects_by_burn_state_adj"):
        assert (out_dir / f"test_{name}.csv").exists(), name
    offs = pd.read_csv(out_dir / "test_neutral_control_offset.csv")
    assert set(offs.columns) >= {"cell", "response", "delta_col", "n",
                                 "n_parents", "offset"}
    assert offs["offset"].abs().min() == pytest.approx(2.0)
    raw = pd.read_csv(out_dir / "test_effects_pooled.csv")
    adj = pd.read_csv(out_dir / "test_effects_pooled_adj.csv")
    assert not np.allclose(raw["mean_cyclen"], adj["mean_cyclen"])


# --------------------------------------------------------------------------- #
# corpus schema: F_xy + burn_state (results 20260830 s11-3)
# --------------------------------------------------------------------------- #
def test_build_steps_emits_the_fxy_columns_and_the_burn_state(tmp_path) -> None:
    """The wave's PRIMARY response must survive the trip into the corpus."""
    boards = _boards(2, seed=11)
    parent_pat = boards[0]
    pg = M.genome_of(parent_pat)
    child_pat = pack_pattern(
        W.enumerate_single_moves(pg)[0][1].to_pattern())
    rows = []
    for i, (packed, parent) in enumerate(((parent_pat, None),
                                          (child_pat, "p0"))):
        rows.append({
            "record_id": "p0" if parent is None else "c0",
            "parent_record_id": parent, "pattern": packed,
            "case_pair": PAIR, "feed": FEED, "library_id": LIBRARY,
            "campaign": "unit", "dataset": "P", "generator": "unit",
            "converged": True, "f_r": 1.5 - 0.1 * i, "f_q": 2.0,
            "cbc_max": 1200.0, "ao_abs": 0.05, "cyclen": 500.0,
            "node_peak": 1.3, "map_cov": 0.2,
            "f_xy": 1.70 - 0.05 * i,
        })
    steps = M.build_steps(pd.DataFrame(rows), {}, {})
    assert len(steps) == 1
    row = steps.iloc[0]
    for col in M.FXY_SCHEMA_COLUMNS:
        assert col in steps.columns, col
    assert row["parent_f_xy"] == pytest.approx(1.70)
    assert row["child_f_xy"] == pytest.approx(1.65)
    assert row["d_f_xy"] == pytest.approx(-0.05)
    assert bool(row["improved_fxy"]) is True
    assert row["burn_state"] == I.burn_state_class(pg, parent_pat, child_pat)
    assert row["burn_state"] in I.BURN_STATES


def test_build_steps_keeps_the_fxy_columns_when_the_store_has_no_label(
        tmp_path) -> None:
    """Stable column SET: an F_xy-less store yields NaN columns, not missing ones."""
    boards = _boards(2, seed=13)
    pg = M.genome_of(boards[0])
    child_pat = pack_pattern(W.enumerate_single_moves(pg)[0][1].to_pattern())
    rows = [{
        "record_id": rid, "parent_record_id": par, "pattern": packed,
        "case_pair": PAIR, "feed": FEED, "library_id": LIBRARY,
        "campaign": "unit", "dataset": "P", "generator": "unit",
        "converged": True, "f_r": 1.5, "f_q": 2.0, "cbc_max": 1200.0,
        "ao_abs": 0.05, "cyclen": 500.0, "node_peak": 1.3, "map_cov": 0.2,
    } for rid, par, packed in (("p0", None, boards[0]),
                               ("c0", "p0", child_pat))]
    steps = M.build_steps(pd.DataFrame(rows), {}, {})
    for col in ("parent_f_xy", "child_f_xy", "d_f_xy"):
        assert col in steps.columns
        assert steps[col].isna().all()
    assert steps["improved_fxy"].isna().all()
    assert steps["burn_state"].notna().all()


def test_corpus_appenders_name_the_migration_on_a_pre_fxy_corpus(
        tmp_path, monkeypatch):
    """A corpus that predates the columns must say WHICH command fixes it."""
    _plan(tmp_path, SIGNED, iso=False, parents=2, monkeypatch=monkeypatch)
    steps = _steps(tmp_path)
    mined = pd.DataFrame([{
        "lineage_source": SIGNED.lineage_source, "campaign": SIGNED.campaign,
        "parent_record_id": "p1", "child_record_id": "c1",
        "move_class": "fresh_relocate", "fresh_radial_dir": "outward",
        "single_move": True, "d_f_r": -0.02, "d_f_xy": -0.01,
        "burn_state": "once",
    }])
    import ablation_analyze as A
    monkeypatch.setattr(A, "build_wave_steps",
                        lambda campaign, lineage: mined.assign(
                            lineage_source=lineage))
    with pytest.raises(SystemExit, match="backfill-fxy"):
        I.cmd_corpus(_corpus_args(tmp_path, tmp_path / "plan_signed.json",
                                  steps))
