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
    # data/reports/ab2_addendum_S1H_20260817.md).  Every override is intentional.
    #
    # This assertion is the tripwire that keeps the deck HONEST about its own
    # champion: `gate-promote` rewrites `model_dir` and nothing else, so a schema
    # change must be reflected here by hand.  It caught exactly that on 2026-08-17.
    model = load_config(EXAMPLE).model
    assert model.cond_schema == "v7"
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
