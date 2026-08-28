"""Serve-time feature/geometry OOD guard (review sec. 4b).

The population-envelope / per-channel z checks are torch-free unit tests; the
backend integration (sidecar persistence + ``feature_ood_types``) uses a tiny
synthetic ensemble and is skipped when torch is unavailable."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from lpopt.data.fuel_types import FuelVec, geom_derived
from lpopt.model.ood_guard import (
    DEFAULT_MARGIN,
    OOD_CHANNELS,
    feature_ood_vecs,
    format_ood_warning,
    population_envelope,
    vec_ood_channels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "data" / "store"


def _lattice_vec(**over) -> FuelVec:
    """A nominal in-population lattice vec (pre-set to the constant geometry)."""
    d = geom_derived(1.285, 0.4096, 0.4750)
    base = dict(
        library_id="5.8_5.1", type_id="X",
        u_avg_enrichment=5.6, enr_main=5.8, enr_zone=5.1, gd_u_enr=4.0,
        u_mass_g=138.8, n_gd=16, gd_wt=8.0,
        kinf0=1.2, kinf10=1.15, kinf20=1.1, kinf30=1.0, bu_k1=42.0,
        ff_pin_max=1.15, boron_worth=-5.5, doppler_coef=-2.0, mtc_dmod=105.0,
        cr1_worth=12400.0, zone_pin_count=52.0,
        xs_a2=0.109, xs_nf2=0.152, xs_s12=0.0167, adf_corner_g2=1.236,
        pin_pitch=1.285, asm_pitch=20.7772, r_pellet=0.4096, r_clad_in=0.4178,
        r_clad_out=0.4750, p_over_d=d["p_over_d"], v_mod_over_v_fuel=d["v_mod_over_v_fuel"],
    )
    base.update(over)
    return FuelVec(**base)


def _population() -> list[FuelVec]:
    # a small spread of in-population vecs so the envelope is non-degenerate on the
    # physics channels but strictly constant (0) on geometry.
    return [
        _lattice_vec(type_id="A", u_mass_g=138.1, xs_s12=0.01644, kinf0=1.18),
        _lattice_vec(type_id="B", u_mass_g=139.45, xs_s12=0.01697, kinf0=1.26),
        _lattice_vec(type_id="C", u_mass_g=138.8, xs_s12=0.0167, kinf0=1.20),
    ]


# --------------------------------------------------------------------------- #
# envelope + per-channel check
# --------------------------------------------------------------------------- #
def test_envelope_geometry_channels_degenerate() -> None:
    env = population_envelope(_population())
    # geometry is constant across the training population -> degenerate [z, z].
    for ch in ("pin_pitch", "asm_pitch", "r_pellet", "r_clad_out", "v_mod_over_v_fuel"):
        assert env[ch][0] == pytest.approx(env[ch][1])
        assert abs(env[ch][0]) < 1e-3
    # physics channels have a real spread.
    assert env["u_mass"][0] < env["u_mass"][1]


def test_in_population_type_not_flagged() -> None:
    env = population_envelope(_population())
    assert vec_ood_channels(_lattice_vec(type_id="C"), env) == []


def test_radius_variant_loud_via_u_mass() -> None:
    """A +1% co-scaled radius variant blows u_mass past the envelope (radius canary)."""
    env = population_envelope(_population())
    base = _lattice_vec()
    rp, ro, ri = 0.4096 * 1.01, 0.4750 * 1.01, 0.4178 * 1.01
    d = geom_derived(1.285, rp, ro)
    var = dataclasses.replace(
        base, r_pellet=rp, r_clad_in=ri, r_clad_out=ro,
        u_mass_g=138.8 * (1.01 ** 2),
        p_over_d=d["p_over_d"], v_mod_over_v_fuel=d["v_mod_over_v_fuel"])
    chans = dict(vec_ood_channels(var, env))
    assert "u_mass" in chans and chans["u_mass"] > 1.0    # loud


def test_pure_pitch_variant_caught_without_u_mass() -> None:
    """A pure pitch change leaves u_mass nominal — the geometry/moderation channels
    must still catch it (the 'silent on the radius canary' case, review sec. 2.3)."""
    env = population_envelope(_population())
    base = _lattice_vec()
    newp = 1.285 * 1.03
    d = geom_derived(newp, 0.4096, 0.4750)
    var = dataclasses.replace(base, pin_pitch=newp, p_over_d=d["p_over_d"],
                              v_mod_over_v_fuel=d["v_mod_over_v_fuel"])
    chans = dict(vec_ood_channels(var, env))
    assert "u_mass" not in chans                          # silent on radius canary
    assert "pin_pitch" in chans or "v_mod_over_v_fuel" in chans


def test_feature_ood_vecs_and_warning() -> None:
    env = population_envelope(_population())
    base = _lattice_vec(type_id="C")
    var = dataclasses.replace(base, type_id="V", pin_pitch=1.285 * 1.03,
                              v_mod_over_v_fuel=1.98)
    flagged = feature_ood_vecs({"C": base, "V": var}, env)
    assert "C" not in flagged and "V" in flagged
    assert "V" in format_ood_warning(flagged)
    assert format_ood_warning({}) == ""


def test_channels_reuse_featurizer_constants() -> None:
    # the u_mass / xs_s12 canaries use the exact featurizer scales.
    from lpopt.model.featurize import _U_MASS_SCALE, _V4_SCALES
    by = {c.name: c for c in OOD_CHANNELS}
    assert by["u_mass"].scale == _U_MASS_SCALE
    assert by["xs_s12"].scale == _V4_SCALES["xs_s12"][1]


# --------------------------------------------------------------------------- #
# backend integration (needs torch + the store)
# --------------------------------------------------------------------------- #
def _make_v4_ensemble(tmp: Path, n: int = 2) -> Path:
    import torch  # noqa: F401
    from lpopt.model.featurize import CHANNELS_V4, FeatureEncoder
    from lpopt.model.net import PosValNet, PosValNetConfig
    from lpopt.model.dataset_torch import TARGETS
    from lpopt.model.train import save_member

    ens = tmp / "ensv4"
    gnames = list(FeatureEncoder(cond_schema="v4").globals_names)
    cfg = PosValNetConfig(in_channels=len(CHANNELS_V4), n_globals=len(gnames))
    zmean = [1.55, 2.3, 1400.0, 690.0, 0.1, 53.0, 70.0]
    zstd = [0.1, 0.1, 60.0, 15.0, 0.05, 1.0, 2.0]
    for i in range(n):
        seed = 700 + i
        save_member(ens / f"member_{seed}", PosValNet(cfg), {
            "net_config": cfg.__dict__, "cond_schema": "v4",
            "channels": list(CHANNELS_V4), "globals": gnames,
            "target_names": list(TARGETS),
            "target_zscore": {"mean": zmean, "std": zstd}, "seed": seed,
        })
    return ens


@pytest.mark.filterwarnings("ignore")
def test_backend_ood_sidecar_and_flagging(tmp_path) -> None:
    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as exc:            # torch DLL flakiness on Windows
        pytest.skip(f"torch unavailable: {exc}")
    if not (STORE / "fuel_types.parquet").is_file():
        pytest.skip("store not present")
    from lpopt.model.model_api import PosValCnnBackend, _FEATURE_OOD_NAME

    ens = _make_v4_ensemble(tmp_path)
    backend = PosValCnnBackend.from_dir(ens, store_dir=STORE, library_id="ga80")

    # envelope computable + covers the guarded channels.
    env = backend.feature_ood_envelope()
    assert "u_mass" in env and "pin_pitch" in env

    # save writes the sidecar; reload restores the SAME frozen envelope.
    out = backend.save(tmp_path / "champ")
    assert (out / _FEATURE_OOD_NAME).is_file()
    reloaded = PosValCnnBackend.load(out, store_dir=STORE, library_id="ga80")
    assert reloaded._feature_ood_envelope == env      # frozen sidecar, not recomputed

    # inject a geometry variant keyed by a short probe alias + build a same-type
    # pattern on it; the guard must flag it while a real ga80 pattern stays clean.
    import pandas as pd
    from lpopt.data.fuel_types import FuelLibrary
    from lpopt.design.geomcheck import build_probe_patterns, side_table_vec
    from lpopt.vendor.masterrl.domain import CaseKey
    import random

    var_row = {
        "library_id": "geomchk", "type_id": "GV0", "probe_key": "GV0",
        "u_mass_g": 138.8 * 1.06, "pin_pitch": 1.285, "asm_pitch": 20.7772,
        "r_pellet": 0.4096 * 1.03, "r_clad_in": 0.4178 * 1.03,
        "r_clad_out": 0.4750 * 1.03,
        **geom_derived(1.285, 0.4096 * 1.03, 0.4750 * 1.03),
    }
    frozen = backend.feature_ood_envelope()           # freeze over base pop first
    extra = pd.DataFrame([side_table_vec(var_row).to_record()])
    backend.fuel = FuelLibrary(pd.concat([backend.fuel.frame, extra], ignore_index=True))
    backend._roster_cache = None
    backend.library_id = "geomchk"
    backend._feature_ood_envelope = frozen            # keep the frozen envelope

    pats = build_probe_patterns("GV0_GV0", 121, 4, random.Random(0))
    flagged = backend.feature_ood_types(pats[0])
    assert "GV0" in flagged
    assert backend.feature_ood_warning(pats) != ""
