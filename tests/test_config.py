"""Config loader: example-deck round-trip, defaults, and unknown-key rejection."""

from __future__ import annotations

from pathlib import Path

import pytest

from lpopt.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "lpopt.inp"


def _fmt_scalar(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"unsupported scalar {type(value)}")  # pragma: no cover


def _fmt_value(value) -> str:
    # Recursive so nested lists (curriculum.e_core_bands) and dict-of-lists
    # (curriculum.cell_pairs) round-trip.  Inline-table keys are quoted so keys
    # containing dots/dashes (e.g. "5.25-5.5_f117") stay a single key.
    if isinstance(value, dict):  # inline table (e.g. produce.strata.generators)
        return "{ " + ", ".join(f'"{k}" = {_fmt_value(v)}' for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_value(x) for x in value) + "]"
    return _fmt_scalar(value)


def _to_toml(sections: dict) -> str:
    """Minimal TOML serializer for round-trip testing.

    Handles scalars, ``list`` (str/float/int), and — for ``[produce]`` — its
    ``strata`` array of tables (each with an inline-table ``generators``).
    """
    lines: list[str] = []
    for section, table in sections.items():
        strata = None
        if section == "produce":
            strata = table.get("strata")
        lines.append(f"[{section}]")
        for key, value in table.items():
            if value is None or key == "strata":
                continue
            lines.append(f"{key} = {_fmt_value(value)}")
        lines.append("")
        for entry in strata or []:
            lines.append("[[produce.strata]]")
            for key, value in entry.items():
                if value is None:
                    continue
                lines.append(f"{key} = {_fmt_value(value)}")
            lines.append("")
    return "\n".join(lines)


def test_load_example_deck() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.remote.host == "HOST_238"
    assert cfg.remote.user == "USER"
    assert cfg.remote.port == 8022
    assert cfg.remote.workdir == "~/lpopt_ws"
    assert cfg.remote.env == "~/lpopt_ws/venv"
    # gpu is a volatile policy value (GPU 1 on 2026-07-17 -> GPU 0 on 2026-07-24 per
    # the user directive "GPU 1 사용 금지, GPU 0 전용"); assert the loader surfaces
    # whatever the deck pins rather than hard-coding a value a re-assignment breaks.
    import tomllib
    deck_gpu = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))["remote"]["gpu"]
    assert cfg.remote.gpu == deck_gpu
    assert cfg.case.mode == "fixed"
    assert cfg.case.pair == "K1_K2"
    assert cfg.case.feed == 121
    assert cfg.master.executable.endswith("master4.0m4_r1.exe")
    assert cfg.verify.package_root == "../3_GA_Surrogate/FEASIBLE_PACKAGE"


def test_roundtrip_example_deck(tmp_path) -> None:
    cfg1 = load_config(EXAMPLE)
    deck2 = tmp_path / "roundtrip.inp"
    deck2.write_text(_to_toml(cfg1.to_dict()), encoding="utf-8")
    cfg2 = load_config(deck2)
    assert cfg1.to_dict() == cfg2.to_dict()


def test_unknown_key_rejected(tmp_path) -> None:
    deck = tmp_path / "bad_key.inp"
    deck.write_text('[remote]\nhosst = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "hosst" in str(exc.value)


def test_unknown_section_rejected(tmp_path) -> None:
    deck = tmp_path / "bad_section.inp"
    deck.write_text("[bogus]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "bogus" in str(exc.value)


def test_defaults_applied(tmp_path) -> None:
    deck = tmp_path / "minimal.inp"
    deck.write_text("[case]\nfeed = 105\n", encoding="utf-8")
    cfg = load_config(deck)
    assert cfg.case.feed == 105
    assert cfg.case.mode == "fixed"  # default
    assert cfg.remote.host == "HOST_238"  # default per plan 4.7
    assert cfg.master.max_cycles == 16  # default


# --------------------------------------------------------------------------- #
# [model] inference selector (plan 4.7): local_cpu | remote_gpu
# --------------------------------------------------------------------------- #
def test_inference_default_unset_is_local(tmp_path) -> None:
    """Absent `inference` == unset ("") so the legacy remote_screening governs."""
    from lpopt.search.campaign import _resolve_inference_mode

    deck = tmp_path / "no_inference.inp"
    deck.write_text("[model]\nbackend = \"posval_cnn\"\n", encoding="utf-8")
    cfg = load_config(deck)
    assert cfg.model.inference == ""                     # unset
    assert _resolve_inference_mode(cfg.model) == "off"   # -> local CPU today


def test_inference_example_deck_is_local_cpu() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.model.inference == "local_cpu"


def test_inference_remote_gpu_resolves_to_auto(tmp_path) -> None:
    from lpopt.search.campaign import _resolve_inference_mode

    deck = tmp_path / "remote.inp"
    deck.write_text("[model]\ninference = \"remote_gpu\"\n", encoding="utf-8")
    cfg = load_config(deck)
    # remote_gpu probes-then-attaches (survives an outage) -> "auto".
    assert _resolve_inference_mode(cfg.model) == "auto"


def test_inference_overrides_legacy_remote_screening(tmp_path) -> None:
    """Explicit local_cpu wins even if the legacy key asks for remote."""
    from lpopt.search.campaign import _resolve_inference_mode

    deck = tmp_path / "conflict.inp"
    deck.write_text(
        "[model]\ninference = \"local_cpu\"\nremote_screening = true\n",
        encoding="utf-8")
    cfg = load_config(deck)
    assert _resolve_inference_mode(cfg.model) == "off"


def test_inference_unset_defers_to_remote_screening(tmp_path) -> None:
    """No `inference` key: behaviour is EXACTLY the legacy remote_screening."""
    from lpopt.search.campaign import _resolve_inference_mode

    deck = tmp_path / "legacy.inp"
    deck.write_text("[model]\nremote_screening = \"auto\"\n", encoding="utf-8")
    cfg = load_config(deck)
    assert cfg.model.inference == ""
    assert _resolve_inference_mode(cfg.model) == "auto"


def test_inference_invalid_rejected(tmp_path) -> None:
    deck = tmp_path / "bad_inference.inp"
    deck.write_text("[model]\ninference = \"gpu_remote\"\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "inference" in str(exc.value)


def test_missing_deck_errors(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.inp")


# --------------------------------------------------------------------------- #
# [model] cond_schema knob (default v3, validated against featurize schemas)
# --------------------------------------------------------------------------- #
def test_cond_schema_default_is_v3() -> None:
    # The DATACLASS default is the real "do NOT flip" guard: it must stay v3 so a
    # deck that omits the key trains the v3 schema.
    from lpopt.config import ModelConfig
    assert ModelConfig().cond_schema == "v3"
    # The shipped example deck DELIBERATELY overrides.  v5 was adopted 2026-07-21
    # with the v5_distill_w160 champion; raised to v6 on 2026-07-25 when the
    # pre-registered A/B promoted arm A6 (52ch = v5 + local-contrast + power-prior
    # channels); raised to v7 on 2026-08-17 when round 12 (arm S1h) promoted the
    # first checkpoint able to encode a 3-fresh-type core (62ch + 18 globals,
    # data/reports/ab2_addendum_S1H_20260817.md); corrected to v8 on 2026-08-29 —
    # the deck had drifted, `data/models/s1i/*/meta.json` says v8 (62ch + 20
    # globals, the composition block widened 3 -> 5 fraction slots).  Every
    # override is intentional.
    #
    # This assertion is the tripwire that keeps the deck HONEST about its own
    # champion: `gate-promote` rewrites `model_dir` and nothing else, so a schema
    # change must be reflected here by hand.  It caught exactly that on 2026-08-17.
    model = load_config(EXAMPLE).model
    assert model.cond_schema == "v8"
    # ... and the map path MUST be able to read the extra channels.  A v6 schema
    # behind the legacy linear map head pays for the channels and cannot use them,
    # which is strictly worse than staying on v5 -- the two settings travel as a
    # set (see config.ModelSection).
    assert model.map_head_mode == "multiscale" or model.map_prior_residual


def test_cond_schema_override(tmp_path) -> None:
    deck = tmp_path / "v4.inp"
    deck.write_text('[model]\ncond_schema = "v4"\n', encoding="utf-8")
    assert load_config(deck).model.cond_schema == "v4"


def test_censor_dataset_a_pin_labels_default_true() -> None:
    # The forensic censoring of Dataset-A's max_pin_burnup surrogate labels is ON
    # by default (data/reports/pinbu_forensics.md); a deck may opt out.
    from lpopt.config import ModelConfig
    assert ModelConfig().censor_dataset_a_pin_labels is True


def test_censor_dataset_a_pin_labels_override(tmp_path) -> None:
    deck = tmp_path / "censor.inp"
    deck.write_text("[model]\ncensor_dataset_a_pin_labels = false\n", encoding="utf-8")
    assert load_config(deck).model.censor_dataset_a_pin_labels is False


def test_promote_fxy_defaults_off_and_a_deck_can_flip_it(tmp_path) -> None:
    """The F_xy head is opt-in (data/reports/fxy_head_prereg_20260829.md): a deck
    that does not mention it must train exactly today's target inventory."""
    from lpopt.config import ModelConfig
    assert ModelConfig().promote_fxy is False

    deck = tmp_path / "fxy.inp"
    deck.write_text("[model]\npromote_fxy = true\n", encoding="utf-8")
    assert load_config(deck).model.promote_fxy is True


def test_cond_schema_invalid_rejected(tmp_path) -> None:
    deck = tmp_path / "bad_schema.inp"
    deck.write_text('[model]\ncond_schema = "v9"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "cond_schema" in str(exc.value)


def test_cond_schema_matches_featurize_inventory() -> None:
    from lpopt.config import _valid_cond_schemas
    from lpopt.model.featurize import CHANNELS_BY_SCHEMA
    assert _valid_cond_schemas() == set(CHANNELS_BY_SCHEMA)


# --------------------------------------------------------------------------- #
# [curriculum] gate + negative-transfer mitigation knobs
# --------------------------------------------------------------------------- #
def test_gate_noreg_epsilon_default_is_recalibrated() -> None:
    """The honest no-regression gate epsilon was recalibrated 0.05 -> 0.10
    (5% family-wise at n=30, max-of-6).  The default MUST be 0.10, and the
    example deck pins the calibrated value."""
    from lpopt.config import CurriculumConfig
    assert CurriculumConfig().gate_noreg_epsilon == pytest.approx(0.10)
    assert load_config(EXAMPLE).curriculum.gate_noreg_epsilon == pytest.approx(0.10)


def test_fr_guard_activation_switch_defaults_off(tmp_path) -> None:
    """D11 (2026-07-26): the no-regression gate's F_r axis is scored but NOT
    enforced by default.  The switch exists so the FA-optimized phase can restore
    the guard with ONE deck setting — and it must ship OFF, because none of
    ``curriculum.FR_GUARD_ACTIVATION_CRITERIA`` is met today (the gate's holdout
    carries 0 sub-1.55 labels; the in-band label ceiling is 0.839 < 0.95)."""
    from lpopt.config import CurriculumConfig
    assert CurriculumConfig().gate_noreg_fr_guard_enabled is False
    assert load_config(EXAMPLE).curriculum.gate_noreg_fr_guard_enabled is False

    deck = tmp_path / "d.inp"
    deck.write_text("[curriculum]\ngate_noreg_fr_guard_enabled = true\n",
                    encoding="utf-8")
    assert load_config(deck).curriculum.gate_noreg_fr_guard_enabled is True


def test_gate_noreg_epsilon_override(tmp_path) -> None:
    deck = tmp_path / "eps.inp"
    deck.write_text("[curriculum]\ngate_noreg_epsilon = 0.07\n", encoding="utf-8")
    assert load_config(deck).curriculum.gate_noreg_epsilon == pytest.approx(0.07)


def test_curriculum_cell_weight_cap_default_and_override(tmp_path) -> None:
    """The curriculum-cell sampling-weight cap OVERRIDE defaults to 16.0 and is a
    settable knob (threaded to the trainer via the retrain split manifest)."""
    from lpopt.config import CurriculumConfig
    assert CurriculumConfig().cell_weight_cap == pytest.approx(16.0)
    assert load_config(EXAMPLE).curriculum.cell_weight_cap == pytest.approx(16.0)
    deck = tmp_path / "cap.inp"
    deck.write_text("[curriculum]\ncell_weight_cap = 12.0\n", encoding="utf-8")
    assert load_config(deck).curriculum.cell_weight_cap == pytest.approx(12.0)


def test_gate_min_f_r_is_retired_not_a_live_looking_dead_knob(tmp_path) -> None:
    """LOW: after the mini campaign moved to the flatness objective, nothing read
    ``gate_min_f_r`` any more — it sat in ``CurriculumConfig`` looking settable
    while controlling nothing, which is the worst of the three options.

    Evidence for REMOVING rather than deprecating in place: a repo-wide search
    over 2,021 ``*.inp`` decks finds zero that set it, so no archived deck breaks.
    The D2 licensing constant it used to carry lives on as
    ``curriculum.FR_GUARD_LICENSING_LIMIT`` (a module constant, not a knob),
    which is what a licensing limit is.
    """
    from lpopt.config import RETIRED_KEYS, CurriculumConfig
    from lpopt.curriculum import FR_GUARD_LICENSING_LIMIT

    assert not hasattr(CurriculumConfig(), "gate_min_f_r")
    assert FR_GUARD_LICENSING_LIMIT == pytest.approx(1.55)

    # A deck that still sets it gets a RETIRED message that says what replaced
    # it, not a bare "typo?" — the deck is wrong, but it is not a typo.
    assert ("curriculum", "gate_min_f_r") in RETIRED_KEYS
    deck = tmp_path / "retired.inp"
    deck.write_text("[curriculum]\ngate_min_f_r = 1.55\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    msg = str(exc.value)
    assert "gate_min_f_r" in msg and "RETIRED" in msg
    assert "typo" not in msg


def test_old_gate_spearman_epsilon_key_is_rejected(tmp_path) -> None:
    """Regression guard: the renamed-away key must not silently no-op — the
    unknown-key discipline catches a stale deck still using the old name."""
    deck = tmp_path / "old.inp"
    deck.write_text("[curriculum]\ngate_spearman_epsilon = 0.05\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "gate_spearman_epsilon" in str(exc.value)


# --------------------------------------------------------------------------- #
# F_xy objective switch (user decision 2026-08-29) — new keys + objective name
# --------------------------------------------------------------------------- #
def test_min_fxy_is_an_accepted_objective(tmp_path) -> None:
    deck = tmp_path / "minfxy.inp"
    deck.write_text('[acquisition]\nobjective = "min_fxy"\n', encoding="utf-8")
    cfg = load_config(deck)
    assert cfg.acquisition.objective == "min_fxy"


def test_an_unknown_objective_still_names_min_fxy_in_the_error(tmp_path) -> None:
    """The validator lists what it accepts; a new mode that is not in that list
    is a mode nobody can discover from the error message."""
    deck = tmp_path / "bad.inp"
    deck.write_text('[acquisition]\nobjective = "min_f_xy"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "min_fxy" in str(exc.value)


def test_fxy_limits_default_and_parse(tmp_path) -> None:
    """``f_xy_limit`` lives in [acquisition] next to ``f_r_limit`` — NOT in
    [constraints], which is the SDM/MTC knob surface."""
    from lpopt.config import AcquisitionConfig

    acq = AcquisitionConfig()
    assert acq.f_xy_limit == pytest.approx(1.65)
    assert acq.minfxy_lambda == pytest.approx(1000.0)          # mirrors minfr_lambda
    assert acq.minfxy_pin_bu_limit == pytest.approx(78.0)      # mirrors minfr pin gate
    assert acq.minfxy_cyclen_lo is None and acq.minfxy_cyclen_hi is None
    assert acq.flatpower_fxy_limit == pytest.approx(1.65)

    deck = tmp_path / "knobs.inp"
    deck.write_text(
        '[acquisition]\n'
        'objective = "min_fxy"\n'
        'f_xy_limit = 1.60\n'
        'minfxy_lambda = 500.0\n'
        'minfxy_pin_bu_limit = 80.0\n'
        'minfxy_cyclen_lo = 615.0\n'
        'minfxy_cyclen_hi = 635.0\n'
        'flatpower_fxy_limit = 1.70\n',
        encoding="utf-8")
    a = load_config(deck).acquisition
    assert a.f_xy_limit == pytest.approx(1.60)
    assert a.minfxy_lambda == pytest.approx(500.0)
    assert a.minfxy_pin_bu_limit == pytest.approx(80.0)
    assert a.minfxy_cyclen_lo == pytest.approx(615.0)
    assert a.minfxy_cyclen_hi == pytest.approx(635.0)
    assert a.flatpower_fxy_limit == pytest.approx(1.70)


def test_f_xy_limit_in_the_constraints_section_is_rejected(tmp_path) -> None:
    """The task instruction spelled it ``[constraints] f_xy_limit``; that section
    is SDM/MTC-only, so the deck must FAIL rather than silently ignore the key."""
    deck = tmp_path / "wrong_section.inp"
    deck.write_text("[constraints]\nf_xy_limit = 1.65\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(deck)
    assert "f_xy_limit" in str(exc.value)


def test_the_min_fxy_deck_example_parses(tmp_path) -> None:
    deck = REPO_ROOT / "fpcamp_minfxy_T6T4_f121_199.inp"
    if not deck.exists():                                   # pragma: no cover
        pytest.skip("deck example not present")
    cfg = load_config(deck)
    assert cfg.acquisition.objective == "min_fxy"
    assert cfg.acquisition.f_xy_limit == pytest.approx(1.65)
    assert cfg.acquisition.f_r_limit == pytest.approx(1.55)   # STILL a constraint
    assert cfg.acquisition.cbc_limit == pytest.approx(1600.0)
    # harvest_maps is what forces keep_success, which is the only reason the
    # final cycle's MAS_OUT (and so f_xy) survives.
    assert cfg.verify.harvest_maps is True


# --------------------------------------------------------------------------- #
# [acquisition] policy_prior — the serving modes and the fail-closed switch
# --------------------------------------------------------------------------- #
def _policy_deck(tmp_path: Path, name: str, body: str) -> Path:
    deck = tmp_path / f"{name}.inp"
    deck.write_text(f'[case]\npair = "K1_K2"\n\n[acquisition]\n{body}',
                    encoding="utf-8")
    return deck


@pytest.mark.parametrize(
    "mode", ["off", "fr", "flat", "both", "v1", "v2", "shadow_v2"])
def test_policy_prior_accepts_every_serving_mode(tmp_path, mode: str) -> None:
    cfg = load_config(_policy_deck(tmp_path, mode, f'policy_prior = "{mode}"\n'))
    assert cfg.acquisition.policy_prior == mode


@pytest.mark.parametrize("mode", ["v3", "shadow", "shadow_v1", "argmax", "v2_fr"])
def test_policy_prior_rejects_anything_else(tmp_path, mode: str) -> None:
    with pytest.raises(ConfigError, match="policy_prior"):
        load_config(_policy_deck(tmp_path, "bad", f'policy_prior = "{mode}"\n'))


def test_policy_prior_case_and_space_tolerance_matches_the_dispatcher(
        tmp_path) -> None:
    """Both sides normalize with ``.strip().lower()``; neither may stop.

    A deck the validator accepts as ``"V2 "`` must reach the dispatcher as
    ``v2``, or the readout and the deck would disagree about which arm ran.
    """

    from lpopt.search.construct import POLICY_MODES

    cfg = load_config(_policy_deck(tmp_path, "loud", 'policy_prior = "V2 "\n'))
    assert cfg.acquisition.policy_prior == "V2 "        # stored verbatim
    assert POLICY_MODES[cfg.acquisition.policy_prior.strip().lower()] == ("v2", "both")


def test_policy_prior_accepted_values_match_the_dispatcher() -> None:
    """The deck validator and ``construct`` must not be able to disagree.

    ``config`` cannot import ``construct`` at module scope (``lpopt.search``
    imports ``config``), so the two lists are written twice — and this is the
    test that keeps the second copy honest.
    """

    from lpopt.config import _VALID_POLICY_PRIORS
    from lpopt.search.construct import POLICY_MODES

    assert set(POLICY_MODES) == _VALID_POLICY_PRIORS


def test_policy_v2_knobs_default_and_round_trip(tmp_path) -> None:
    from lpopt.config import AcquisitionConfig

    default = AcquisitionConfig()
    assert default.policy_prior_model_dir_v2 == "data/models/policy_v2"
    assert default.policy_prior_strict is False
    # NOT v1's 0.25: v2's output spread is ~3x narrower, so v1's temperature
    # would make the softmax nearly uniform (policy_v2_results section 8).
    assert default.policy_prior_temperature_v2 == pytest.approx(0.08)
    assert default.policy_prior_temperature == pytest.approx(0.25)

    cfg = load_config(_policy_deck(
        tmp_path, "prod",
        'policy_prior = "v2"\n'
        'policy_prior_model_dir_v2 = "data/models/policy_v2_runA"\n'
        'policy_prior_strict = true\n'
        'policy_prior_temperature_v2 = 0.06\n'))
    acq = cfg.acquisition
    assert acq.policy_prior == "v2"
    assert acq.policy_prior_model_dir_v2 == "data/models/policy_v2_runA"
    assert acq.policy_prior_strict is True
    assert acq.policy_prior_temperature_v2 == pytest.approx(0.06)
    # the v1 knobs are untouched by a v2 deck
    assert acq.policy_prior_model_dir == "data/models/policy_v1"
