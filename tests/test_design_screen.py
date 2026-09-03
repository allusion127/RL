"""Task #4/#4a/#6 regression: the DeCART lattice-surrogate bridge.

Runs with **no** surrogate installed: the bridge must import, the mapping
assertions must hold, and the enumerate/dedup/gate functions are pure.  A stub
surrogate tree is synthesised in ``tmp_path`` to exercise the load path and the
``predict()`` contract without torch or 196 MB of checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpopt.design import screen as S
from lpopt.design.spec import FuelDesign

REPO_ROOT = Path(__file__).resolve().parents[1]
APR1400 = (REPO_ROOT / ".." / "0_APR1400").resolve()


# --------------------------------------------------------------------------- #
# 1.  the z1 <-> PB / z2 <-> PA mapping                                (R17)
# --------------------------------------------------------------------------- #
def test_zoning_mapping_is_the_counter_intuitive_one() -> None:
    assert S.Z_TO_PATTERN == {"z1": "PB", "z2": "PA"}
    assert S.pattern_for("z1") == "PB"
    assert S.pattern_for("z2") == "PA"
    assert S.PATTERN_TO_Z == {"PB": "z1", "PA": "z2"}


def test_zoning_mapping_is_derived_from_the_template_octant_rows() -> None:
    """dec_FA_A01.inp:97 = '1 1 1 1 1 1 2 2' (PB); A02:97 = all 2s (PA)."""
    assert S._row7_pattern((1, 1, 1, 1, 1, 1, 2, 2)) == "PB"
    assert S._row7_pattern((2, 2, 2, 2, 2, 2, 2, 2)) == "PA"
    assert S._OCTANT_ROW7["z1"] == (1, 1, 1, 1, 1, 1, 2, 2)
    assert S._OCTANT_ROW7["z2"] == (2, 2, 2, 2, 2, 2, 2, 2)
    with pytest.raises(ValueError):
        S._row7_pattern((1, 2, 1, 2, 1, 2, 1, 2))


def test_pattern_for_rejects_a_forced_literal() -> None:
    """Passing 'PA' where a zoning_variant belongs must raise, not silently
    screen the wrong lattice family (MOCHA's default is surrogate_pattern=PA)."""
    with pytest.raises(ValueError):
        S.pattern_for("PA")
    with pytest.raises(ValueError):
        S.pattern_for("z3")


@pytest.mark.skipif(not (APR1400 / "5.8_5.1" / "FA" / "IGD_16" / "6_16_z1"
                         / "dec_FA_A01.inp").is_file(),
                    reason="0_APR1400 templates not staged on this host")
def test_zoning_mapping_matches_the_decks_on_disk() -> None:
    assert S.verify_zoning_against_templates(APR1400) == {"z1": "PB", "z2": "PA"}


def test_build_assembly_rows_differs_only_in_row_7() -> None:
    layout = "1:1;4:1;6:4"
    pa = S.build_assembly_rows("PA", layout).split("|")
    pb = S.build_assembly_rows("PB", layout).split("|")
    assert pa[:7] == pb[:7]
    assert pa[7] == "2 2 2 2 2 2 2 2"
    assert pb[7] == "1 1 1 1 1 1 2 2"
    # guide tubes untouched, Gd where asked
    assert pa[0] == "9"
    assert pa[1].split() == ["2", "3"]          # (1,0) zoning, (1,1) Gd


# --------------------------------------------------------------------------- #
# 2.  bounds enforced, grid not
# --------------------------------------------------------------------------- #
def _row(**kw):
    base = dict(u_high=5.50, du=0.75, u_low=4.75, gd_u=4.0, gd_wt=8, n_gd=20,
                gd_positions="1:1;4:1;6:4", pattern="PB")
    base.update(kw)
    # The surrogate feature builders require this key on every row
    # (features.py:145,341), so a realistic row carries it.
    base.setdefault("assembly_rows",
                    S.build_assembly_rows(base["pattern"], base["gd_positions"]))
    return base


def test_off_grid_du_passes() -> None:
    """T3/T4 were screened at du = 0.75 -- off the 0.1 grid -- and cleared the
    DeCART cross-check at < 100 pcm.  Rejecting off-grid would disqualify them."""
    assert S.validate_bounds(_row(du=0.75)) == []


def test_out_of_bounds_du_fails() -> None:
    """OPSCREEN's u5.50/4.6750 is du = 0.825, past the 0.80 bound."""
    errs = S.validate_bounds(_row(du=0.825, u_low=4.675))
    assert errs and any("du" in e and "outside bounds" in e for e in errs)


def test_bounds_on_every_axis() -> None:
    assert any("u_high" in e for e in S.validate_bounds(_row(u_high=7.05)))
    assert any("gd_u" in e for e in S.validate_bounds(_row(gd_u=4.5)))
    assert any("gd_wt" in e for e in S.validate_bounds(_row(gd_wt=7)))
    assert any("n_gd" in e for e in S.validate_bounds(_row(n_gd=28)))
    assert any("n_gd" in e for e in S.validate_bounds(_row(n_gd=0)))
    assert S.validate_bounds(_row(u_high=7.00, du=0.80, u_low=6.20)) == []


def test_layout_rules() -> None:
    # census mismatch
    assert any("multiplicity" in e
               for e in S.validate_layout([(1, 1), (4, 1)], 20))
    # (1,1) diag 4 + (4,1) 8 + (6,4) 8 = 20
    assert S.validate_layout([(1, 1), (4, 1), (6, 4)], 20) == []
    # Chebyshev
    assert any("Chebyshev" in e
               for e in S.validate_layout([(4, 1), (5, 1), (6, 4)], 20))
    # not a candidate position
    assert any("candidate" in e
               for e in S.validate_layout([(0, 0), (4, 1), (6, 4)], 20))
    # duplicates
    assert any("duplicate" in e
               for e in S.validate_layout([(4, 1), (4, 1), (2, 0)], 24))


# --------------------------------------------------------------------------- #
# 3.  FuelDesign -> surrogate row
# --------------------------------------------------------------------------- #
def test_design_to_row_uses_the_mapping_and_physical_units() -> None:
    d = FuelDesign(5.50, 4.70, "z1", 8.0, 20)
    row = S.design_to_row(d, "1:1;4:1;6:4")
    assert row["u_high"] == 5.50
    assert row["u_low"] == 4.70
    assert row["du"] == pytest.approx(0.80)
    assert row["gd_u"] == 4.0
    assert row["gd_wt"] == 8 and row["n_gd"] == 20
    assert row["gd_positions"] == "1:1;4:1;6:4"
    assert row["pattern"] == "PB"          # z1, not PA
    assert row["zoning"] == "z1"
    assert row["assembly_rows"].split("|")[7] == "1 1 1 1 1 1 2 2"
    assert S.validate_bounds(row) == []


def test_every_row_handed_to_the_surrogate_carries_assembly_rows() -> None:
    """Regression: ``assembly_rows`` used to be opt-in, so the canonical slice-Z
    row passed ``validate_bounds`` and then raised ``KeyError`` inside
    ``features.py:145/341`` after torch and the checkpoints had loaded.  The
    reconstruction that would have saved it lives in ``predict.py:245
    validate_design``, which this module bypasses to admit off-grid ``du``."""
    d = FuelDesign(5.50, 4.70, "z1", 8.0, 20)
    row = S.design_to_row(d, "1:1;4:1;6:4")
    assert "assembly_rows" in row
    assert row["assembly_rows"] == S.build_assembly_rows("PB", "1:1;4:1;6:4")
    assert row["assembly_rows"].split("|")[7] == "1 1 1 1 1 1 2 2"
    assert S.validate_bounds(row) == []


def test_validate_bounds_rejects_a_row_without_assembly_rows() -> None:
    row = _row()
    del row["assembly_rows"]
    errs = S.validate_bounds(row)
    assert errs and any("assembly_rows" in e for e in errs)
    # the numeric axes alone can still be checked deliberately
    assert S.validate_bounds(row, require_assembly_rows=False) == []


def test_validate_bounds_rejects_assembly_rows_that_disagree() -> None:
    """A stale map (right shape, wrong pattern) is worse than a missing one."""
    row = _row(pattern="PB")
    row["assembly_rows"] = S.build_assembly_rows("PA", "1:1;4:1;6:4")
    errs = S.validate_bounds(row)
    assert errs and any("disagrees" in e for e in errs)


def test_bridge_fills_assembly_rows_for_hand_built_rows(tmp_path: Path) -> None:
    """Rows also arrive from a catalog or a CSV, so the bridge fills the key
    itself rather than trusting the caller -- without mutating the caller's dict."""
    b = S.SurrogateBridge(_stub_tree(tmp_path))
    row = _row()
    del row["assembly_rows"]
    out = b.predict([row])
    assert out["rows"][0]["assembly_rows"] == S.build_assembly_rows(
        "PB", "1:1;4:1;6:4")
    assert "assembly_rows" not in row          # caller's dict untouched
    one = b.predict_one(dict(row))
    assert one["feat"][0] == out["rows"][0]["assembly_rows"]


def test_design_to_row_requires_a_layout() -> None:
    d = FuelDesign(5.50, 4.70, "z1", 8.0, 20)
    with pytest.raises(ValueError, match="gd_positions is required"):
        S.design_to_row(d)


def test_design_to_row_accepts_a_mapping() -> None:
    row = S.design_to_row(dict(e1=5.0, e2=4.25, zoning_variant="z2",
                               gd_wt=10, n_gd=20,
                               gd_positions="1:1;4:1;6:4"))
    assert row["pattern"] == "PA"
    assert row["du"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# 4.  catalog
# --------------------------------------------------------------------------- #
def _designs_json(tmp_path: Path) -> Path:
    p = tmp_path / "designs.json"
    p.write_text(json.dumps({
        "library_id": "paramA",
        "designs": [
            {"type_id": "P5042Z1G10N16", "e1": 5.0, "e2": 4.25,
             "zoning_variant": "z1", "gd_wt": 10.0, "n_gd": 16, "alias": "T3",
             "gd_u_enr": 4.0, "gd_positions": "1:1;5:2;5:5",
             "provenance": "realize_lat1600 2026-08-11"},
            {"type_id": "P5849Z1G08N16", "e1": 5.8, "e2": 4.93,
             "zoning_variant": "z1", "gd_wt": 8.0, "n_gd": 16, "alias": "P0",
             "gd_u_enr": 4.0},
        ],
    }), encoding="utf-8")
    return p


def test_catalog_loads_from_designs_json(tmp_path: Path) -> None:
    cat = S.load_design_catalog(_designs_json(tmp_path))
    assert cat.library_id == "paramA"
    assert len(cat) == 2
    t3 = cat["P5042Z1G10N16"]
    assert t3.alias == "T3"
    assert t3.gd_positions == ((1, 1), (5, 2), (5, 5))
    assert t3.pattern == "PB"
    assert cat.by_alias("T3").type_id == "P5042Z1G10N16"
    assert t3.extra["provenance"].startswith("realize_lat1600")


def test_catalog_warns_on_rows_without_layout(tmp_path: Path) -> None:
    cat = S.load_design_catalog(_designs_json(tmp_path))
    assert len(cat.with_layout) == 1
    assert any("P5849Z1G08N16" in w and "no gd_positions" in w
               for w in cat.warnings)
    with pytest.raises(S.NotAvailable, match="no gd_positions"):
        cat["P5849Z1G08N16"].to_row()


def test_catalog_entry_to_row(tmp_path: Path) -> None:
    cat = S.load_design_catalog(_designs_json(tmp_path))
    row = cat["P5042Z1G10N16"].to_row()
    assert row["pattern"] == "PB" and row["n_gd"] == 16
    assert S.validate_bounds(row) == []


def test_catalog_rejects_duplicate_type_id(tmp_path: Path) -> None:
    p = tmp_path / "dup.json"
    d = {"type_id": "X", "e1": 5.0, "e2": 4.25, "zoning_variant": "z1",
         "gd_wt": 8.0, "n_gd": 16, "gd_positions": "1:1;5:2;5:5"}
    p.write_text(json.dumps({"library_id": "x", "designs": [d, dict(d)]}),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate type_id"):
        S.load_design_catalog(p)


# --------------------------------------------------------------------------- #
# 5.  enumeration
# --------------------------------------------------------------------------- #
def test_layout_catalogue_matches_the_surrogate_manifest() -> None:
    """SURROGATE_USAGE.md sec 7.5: 89 valid layouts, n12:18 n16:24 n20:28 n24:19."""
    layouts = S.enumerate_gd_layouts()
    counts = {n: len(v) for n, v in layouts.items()}
    assert counts == {12: 18, 16: 24, 20: 28, 24: 19}
    assert sum(counts.values()) == 89
    # every layout is self-consistent
    for n, combos in layouts.items():
        for combo in combos:
            assert S.validate_layout(combo, n) == []
    # the slice-Z layout is in the n_gd=20 set
    assert ((1, 1), (4, 1), (6, 4)) in layouts[20]


def test_enumerate_round1_is_deterministic_and_explains_itself() -> None:
    a = S.enumerate_round1()
    b = S.enumerate_round1()
    assert a == b
    assert a.n_u_high == 11                     # 5.00..5.50 step 0.05
    assert a.n_gd_wt == 3
    assert a.n_layout_pairs == 89
    assert a.n_zoning == 2
    assert a.count == (a.n_enrichment_pairs * a.n_gd_wt
                       * a.n_layout_pairs * a.n_zoning)
    # v2's 3,738 assumed ratio == 0.85 exactly; the registered window is wider
    assert a.count != 3738


def test_enumerate_round1_enrichment_window() -> None:
    a = S.enumerate_round1()
    for u, e2 in a.enrichment_pairs:
        assert 5.00 - 1e-9 <= u <= 5.50 + 1e-9
        assert 0.82 - 1e-9 <= e2 / u <= 0.88 + 1e-9
        assert 0.40 - 1e-9 <= u - e2 <= 0.80 + 1e-9
        assert abs(round(e2 * 20) / 20 - e2) < 1e-9      # on the 0.05 grid
    # both slice-Z tuples are inside the window
    assert (5.50, 4.70) in a.enrichment_pairs
    assert (5.00, 4.25) in a.enrichment_pairs
    # ... and the rejected OPSCREEN draft (du = 0.825) is not
    assert (5.50, 4.675) not in a.enrichment_pairs
    assert a.n_enrichment_pairs == 43


def test_enumerate_cli(capsys) -> None:
    assert S.main(["enumerate"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["count"] == S.enumerate_round1().count
    assert out["layouts_by_n_gd"] == {"12": 18, "16": 24, "20": 28, "24": 19}


# --------------------------------------------------------------------------- #
# 6.  graceful degradation + the stub surrogate
# --------------------------------------------------------------------------- #
def test_module_imports_without_a_surrogate() -> None:
    """The import above already proves it; assert the contract explicitly."""
    assert S.NotAvailable is not None
    assert issubclass(S.NotAvailable, RuntimeError)


def test_find_surrogate_root_reports_the_precise_blocker(tmp_path: Path) -> None:
    with pytest.raises(S.NotAvailable) as ei:
        S.find_surrogate_root(tmp_path)
    msg = str(ei.value)
    assert "surrogate/predict.py" in msg and "surrogate_runs/ens_peak" in msg
    assert str(tmp_path) in msg


def test_bridge_is_lazy_and_reports_unavailable(tmp_path: Path) -> None:
    b = S.SurrogateBridge(tmp_path)
    assert b.available is False
    assert "no usable DeCART lattice-surrogate tree" in (b.load_error or "")
    with pytest.raises(S.NotAvailable):
        b.predict([_row()])


def _stub_tree(tmp_path: Path) -> Path:
    """A minimal fake surrogate tree: real package layout, fake predictor."""
    root = tmp_path / "kpin_pa"
    (root / "surrogate").mkdir(parents=True)
    ds = root / "surrogate_runs" / "dataset_stub_v1"
    ds.mkdir(parents=True)
    (root / "surrogate_runs" / "ens_peak").mkdir(parents=True)
    (ds / "bu_grid.npy").write_bytes(b"")           # presence is what is checked
    (root / "surrogate" / "predict.py").write_text(
        "import json\n"
        "class Engines:\n"
        "    def __init__(self, args):\n"
        "        self.args = args\n"
        # The fake reads exactly the keys the real feature builders read, so a
        # row missing 'assembly_rows' raises here the way it would upstream
        # (features.py:145 build_case_features, :341 build_batch_features).
        "_KEYS = ('assembly_rows', 'u_high', 'u_low', 'du', 'gd_u',\n"
        "         'gd_wt', 'n_gd', 'pattern')\n"
        "def _feat(row):\n"
        "    return tuple(row[k] for k in _KEYS)\n"
        "def predict_case(eng, row):\n"
        "    return {'peak_max': 1.1208, 'peak_max_bu': 6.0,\n"
        "            'k_bu0': 1.21, 'crossing_bu': 41.0, 'row': row,\n"
        "            'feat': _feat(row)}\n"
        "def predict_cases(eng, rows, fast=False, **kw):\n"
        "    feats = [_feat(r) for r in rows]\n"
        "    n = len(rows)\n"
        "    return {'peak_max': [1.1208] * n, 'peak_max_bu': [6.0] * n,\n"
        "            'k_bu0': [1.21] * n, 'crossing_bu': [41.0] * n,\n"
        "            'rows': rows, 'feats': feats, 'fast': fast}\n",
        encoding="utf-8")
    return root


def test_stub_surrogate_round_trip(tmp_path: Path) -> None:
    root = _stub_tree(tmp_path)
    b = S.SurrogateBridge(root)
    assert b.available is True
    assert b.load_error is None
    out = b.predict([_row(), _row(gd_wt=10, n_gd=20)])
    assert out["rows"][0]["pattern"] == "PB"
    f = S.screen_features(out, 0)
    assert f["ff"] == pytest.approx(1.1208)
    assert f["crossing_bu"] == pytest.approx(41.0)
    one = b.predict_one(_row())
    assert one["peak_max"] == pytest.approx(1.1208)
    # engines are built once
    assert b.engines() is b.engines()


def test_stub_surrogate_rejects_out_of_bounds_rows(tmp_path: Path) -> None:
    b = S.SurrogateBridge(_stub_tree(tmp_path))
    with pytest.raises(ValueError, match="outside bounds"):
        b.predict([_row(n_gd=20, du=0.9, u_low=4.6)])


def test_probe_cli_on_stub(tmp_path: Path, capsys) -> None:
    root = _stub_tree(tmp_path)
    assert S.main(["probe", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["available"] is True
    assert S.main(["probe", "--root", str(tmp_path / "nope")]) == 1
    assert json.loads(capsys.readouterr().out)["available"] is False


# --------------------------------------------------------------------------- #
# 7.  diversity / dedup / role-pair gate                            (task #6)
# --------------------------------------------------------------------------- #
def test_dedup_exact_folds_identical_designs_and_keeps_layout_variants() -> None:
    rows = [_row(), _row(), _row(gd_positions="2:2;4:1;6:3")]
    keep = S.dedup_exact(rows)
    assert keep == [0, 2]
    # same (gd_wt, n_gd, z) but a different layout is NOT a duplicate
    assert S.design_key(rows[0]) != S.design_key(rows[2])


def test_dedup_near_is_deterministic() -> None:
    desc = [[0.0, 0.0], [0.02, 0.0], [1.0, 1.0], [1.01, 0.99], [3.0, -2.0]]
    a = S.dedup_near(desc, tol=0.25)
    b = S.dedup_near(desc, tol=0.25)
    assert a == b
    assert a == [0, 2, 4]
    assert S.dedup_near(desc, tol=0.0) == [0, 1, 2, 3, 4]


def test_z_scale_handles_degenerate_channels() -> None:
    sc = S.z_scale([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    assert sc[1] == 1.0                       # sd == 0 -> scale 1, not a div0
    assert sc[0] > 0.0


def test_greedy_maxmin_deterministic_and_spread() -> None:
    desc = [[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]
    a = S.greedy_maxmin(desc, 3)
    assert a == S.greedy_maxmin(desc, 3)
    assert len(a) == 3 and len(set(a)) == 3
    assert S.greedy_maxmin(desc, 99) == S.greedy_maxmin(desc, 5)
    assert S.greedy_maxmin(desc, 0) == []
    assert S.greedy_maxmin([], 3) == []


def test_role_pair_gate_rejects_low_contrast() -> None:
    v = S.role_pair_gate([(0, 1), (2, 3), (4, 5)], [0.0308, 0.020, 0.026])
    assert [x.ok for x in v] == [True, False, True]
    assert "0.020" in v[1].reason and "0.026" in v[1].reason
    assert v[0].reason == ""


def test_select_pairs_is_deterministic_and_gates_first() -> None:
    desc = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [3.0, 3.0], [4.0, 4.0]]
    pairs = [(0, 1), (2, 3), (4, 5)]
    contrasts = [0.031, 0.020, 0.045]        # middle pair fails the gate
    got = S.select_pairs(desc, pairs, contrasts, k=2)
    assert got == S.select_pairs(desc, pairs, contrasts, k=2)
    assert (2, 3) not in got
    assert set(got) <= {(0, 1), (4, 5)}
    assert S.select_pairs(desc, pairs, [0.0, 0.0, 0.0], k=2) == []


def test_contrast_gate_default_comes_from_opscreen_chain() -> None:
    from lpopt.design.opscreen_chain import CONTRAST_MIN
    v = S.role_pair_gate([(0, 1)], [CONTRAST_MIN])
    assert v[0].ok is True
    v = S.role_pair_gate([(0, 1)], [CONTRAST_MIN - 1e-6])
    assert v[0].ok is False


# --------------------------------------------------------------------------- #
# 8.  need.py skeleton -- the F1 interlock                          (task #7/#9)
# --------------------------------------------------------------------------- #
def test_sigma_ladder_bands() -> None:
    from lpopt.design.need import sigma_ladder_verdict as v
    assert v(0.004).band == "resolving" and v(0.004).k_bar == 2.0
    assert v(0.005).band == "order-only"
    assert v(0.020).band == "order-only"
    assert v(0.0201).band == "unresolving"
    assert v(0.020).triggers_valid is True
    assert v(0.0201).triggers_valid is False


def test_zoning_guards_survive_python_O() -> None:
    """The z1<->PB guard must not be an ``assert``: ``python -O`` compiles those
    out, and the guard would vanish exactly when a bad edit needs catching."""
    src = Path(S.__file__).read_text(encoding="utf-8")
    assert "assert Z_TO_PATTERN" not in src
    assert "assert derived == Z_TO_PATTERN" not in src
    assert 'raise RuntimeError(\n        "z1<->PB' in src


def test_ood_scale_uses_the_registered_envelope(monkeypatch) -> None:
    """The registered diversity scale is the ood_guard population envelope, not
    the candidate set's own sd (which drifts between rounds)."""
    import lpopt.model.ood_guard as og
    monkeypatch.setattr(og, "population_envelope_from_library",
                        lambda fuel, ids=None: {"a": [1.0, 3.0], "b": [2.0, 2.0]})
    assert S.ood_scale(object(), ["a", "b", "missing"]) == [2.0, 1.0, 1.0]
    # the fallback is documented as a fallback
    assert "FALLBACK" in S.z_scale.__doc__.upper()


def test_k_bar_is_the_multiplier_and_bar_is_the_bar() -> None:
    """``k_bar`` used to be ``2.0*s/max(s,1e-12)`` -- identically 2.0, written as
    if it were the bar.  The multiplier and the bar are now separate."""
    from lpopt.design.need import sigma_ladder_verdict as v
    for s in (0.006, 0.010, 0.020):
        assert v(s).k_bar == 2.0
        assert v(s).bar == pytest.approx(2.0 * s)
    assert v(0.004).bar == pytest.approx(0.008)
    assert v(0.0774).k_bar is None and v(0.0774).bar is None


def test_measured_sigma_voids_the_triggers() -> None:
    """The task-#0 retrodiction measured sigma_chain,paired = 0.071-0.083."""
    from lpopt.design.need import LaunchRules, launch_allowed, sigma_ladder_verdict
    for sigma in (0.0713, 0.0774, 0.083):
        assert sigma_ladder_verdict(sigma).triggers_valid is False
        ok, why = launch_allowed(LaunchRules(sigma_chain_paired=sigma))
        assert ok is False and "unresolving" in why


def test_launch_is_refused_without_a_measured_sigma() -> None:
    from lpopt.design.need import LaunchRules, launch_allowed
    ok, why = launch_allowed(LaunchRules())
    assert ok is False
    assert "F1 unmet" in why
    assert "assembly_sigma_chain_retrodiction_20260903.md" in why


def test_need_signal_is_not_faked() -> None:
    from lpopt.design.need import EXCLUDED_CHANNELS, need_signal
    assert "geometry" in EXCLUDED_CHANNELS
    with pytest.raises(NotImplementedError):
        need_signal()
